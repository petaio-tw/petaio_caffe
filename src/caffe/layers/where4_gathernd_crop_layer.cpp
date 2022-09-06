#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/where4_gathernd_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"      



namespace caffe {

template <typename Dtype>
void Where4GatherndCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
	 const Where4GatherndCropParameter& where4_gathernd_crop_param = this->layer_param_.where4_gathernd_crop_param();
  num_output_ = where4_gathernd_crop_param.num_output();
  axis_ = bottom[1]->CanonicalAxisIndex(where4_gathernd_crop_param.axis());
  CHECK_GE(num_output_, 1) << "num_output must not be less than 1.";
  CHECK_GE(axis_, 0) << "axis must not be less than 0.";
  CHECK_LE(axis_, bottom[1]->num_axes()) <<
	"axis must be less than or equal to the number of axis.";
  CHECK_LE(num_output_, bottom[1]->shape(axis_))
	<< "num_output must be less than or equal to the dimension of the axis.";

  CHECK_GT(where4_gathernd_crop_param.crop_h(), 0)
      << "crop_h must be > 0";
  CHECK_GT(where4_gathernd_crop_param.crop_w(), 0)
      << "crop_w must be > 0";
  crop_height_ = where4_gathernd_crop_param.crop_h();
  crop_width_ = where4_gathernd_crop_param.crop_w();
  extrapolation_value_ = where4_gathernd_crop_param.extrapolation_value();
}

template <typename Dtype>
void Where4GatherndCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
	std::vector<int> shape(2);
	shape[0] = num_output_;
	shape[1] = bottom[1]->num_axes();
	indices_shape_.clear();
	for(int i=0;i<shape.size();i++)
	  indices_shape_.push_back(shape[i]);

 	const int num_axes = bottom[0]->num_axes();
 	CHECK_GE(num_axes, 1) << "the dimension of input should be larger than or equal to 1";
 	indices_dim_ = indices_shape_.size();                          
 	CHECK_GE(indices_dim_, 1) << "the dimension of indices should be larger than or equal to 1";
 	int count = 1;
 	for (int i = 0; i < indices_shape_.size(); ++i) {
	  	count *= indices_shape_[i];
	 }
 	vector<int> bottom_shape = bottom[0]->shape();
 	vector<int> top_shape = bottom[0]->shape();
 	indices_N_ = indices_shape_[indices_shape_.size()-1];
 	CHECK_LE(indices_N_, num_axes) << "indices.shape[-1] must be <= params.rank, but saw indices.shape[-1]:" 
	    	<< indices_N_ << ", and params.rank: " << num_axes;
 	top_shape.resize(indices_dim_ - 1 + num_axes - indices_N_);     
 	gather_nd_size_ = bottom[0]->count(indices_N_);            

 	// The result shape is
 	//   indices.shape[:-1] + params.shape[indices.shape[-1]:]
	 for (int i = 0; i < indices_dim_ - 1; ++i) {
		  top_shape[i] = indices_shape_[i];
 	}
	 for (int i = 0; i < num_axes - indices_N_; ++i) {
		  top_shape[i + indices_dim_ - 1] = bottom_shape[i + indices_N_];
 	}
	gather_output_.Reshape(top_shape);

	// input image check
	CHECK_EQ(bottom[2]->num_axes(), 4) << "bottom[2] must have 4 axes.";
	CHECK_EQ(bottom[3]->num_axes(), 4) << "bottom[3] must have 4 axes.";
	CHECK_EQ(bottom[4]->num_axes(), 4) << "bottom[4] must have 4 axes.";
	CHECK_EQ(bottom[5]->num_axes(), 4) << "bottom[5] must have 4 axes.";
	channels_ = bottom[2]->shape(3);
	CHECK_EQ(bottom[2]->shape(3), bottom[3]->shape(3))
		<< "Input images should have equal channel count.";
	CHECK_EQ(bottom[3]->shape(3), bottom[4]->shape(3))
		<< "Input images should have equal channel count.";
	CHECK_EQ(bottom[4]->shape(3), bottom[5]->shape(3))
		<< "Input images should have equal channel count.";
    top[0]->Reshape(num_output_, crop_height_, crop_width_, channels_);
}

template <typename Dtype>
void Where4GatherndCropLayer<Dtype>::Crop_And_Resize(const Dtype *bottom_data, const Dtype *bottom_rois,
		Dtype *top_data, int num_boxes_, int image_height_, int image_width_)
{
    for (int b = 0; b < num_boxes_; ++b)
    {
      const float y1 = bottom_rois[b*4];
      const float x1 = bottom_rois[b*4+1];
      const float y2 = bottom_rois[b*4+2];
      const float x2 = bottom_rois[b*4+3];

      const int b_in = 0; // Assume batch_size is always 1

      const float height_scale =
          (crop_height_ > 1) ? (y2 - y1) * (image_height_ - 1) / (crop_height_ - 1) : 0;
      const float width_scale =
          (crop_width_ > 1) ? (x2 - x1) * (image_width_ - 1) / (crop_width_ - 1) : 0;

      for (int y = 0; y < crop_height_; ++y) {
        const float in_y = (crop_height_ > 1)
                               ? y1 * (image_height_ - 1) + y * height_scale
                               : 0.5 * (y1 + y2) * (image_height_ - 1);
        if (in_y < 0 || in_y > image_height_ - 1) {
          for (int x = 0; x < crop_width_; ++x) {
            for (int d = 0; d < channels_; ++d) {
              top_data[((b*crop_height_+y)*crop_width_+x)*channels_+d] = extrapolation_value_;
              //crops(b, y, x, d) = extrapolation_value;
            }
          }
          continue;
        }

        const int top_y_index = floorf(in_y);
        const int bottom_y_index = ceilf(in_y);
        const float y_lerp = in_y - top_y_index;

        for (int x = 0; x < crop_width_; ++x) {
             const float in_x = (crop_width_ > 1)
                                    ? x1 * (image_width_ - 1) + x * width_scale
                                    : 0.5 * (x1 + x2) * (image_width_ - 1);
             if (in_x < 0 || in_x > image_width_ - 1) {
               for (int d = 0; d < channels_; ++d) {
            	 top_data[((b*crop_height_+y)*crop_width_+x)*channels_+d] = extrapolation_value_;
                 //crops(b, y, x, d) = extrapolation_value;
               }
               continue;
             }
             const int left_x_index = floorf(in_x);
             const int right_x_index = ceilf(in_x);
             const float x_lerp = in_x - left_x_index;

             for (int d = 0; d < channels_; ++d) {
               int index = ((b_in*image_height_+top_y_index)*image_width_+left_x_index)*channels_+d;
               //(image(b_in, top_y_index, left_x_index, d));
               const float top_left = static_cast<float>(bottom_data[index]);

               index = ((b_in*image_height_+top_y_index)*image_width_+right_x_index)*channels_+d;
               //(image(b_in, top_y_index, right_x_index, d));
               const float top_right = static_cast<float>(bottom_data[index]);

               index = ((b_in*image_height_+bottom_y_index)*image_width_+left_x_index)*channels_+d;
               //(image(b_in, bottom_y_index, left_x_index, d));
               const float bottom_left = static_cast<float>(bottom_data[index]);

               index = ((b_in*image_height_+bottom_y_index)*image_width_+right_x_index)*channels_+d;
               //(image(b_in, bottom_y_index, right_x_index, d));
               const float bottom_right = static_cast<float>(bottom_data[index]);

               const float top = top_left + (top_right - top_left) * x_lerp;
               const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

               top_data[((b*crop_height_+y)*crop_width_+x)*channels_+d] = top + (bottom - top) * y_lerp;
               //crops(b, y, x, d) = top + (bottom - top) * y_lerp;

             }
           }
       }
    }
}

template <typename Dtype>
void Where4GatherndCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	   const vector<Blob<Dtype>*>& top) {
  indices_.clear();
  const Dtype* indices_data = bottom[1]->cpu_data();
  vector<int> group, group1, group2, group3;
  // TODO: add handling for other dimension conditions
  if (bottom[1]->num_axes() == 2)
  {
	if (axis_ == 1)
	{
	  for (int i=0; i<bottom[1]->shape(0); i++)
	  {
		for (int j=0; j<bottom[1]->shape(1);j++)
		{
			int value = int(indices_data[i*bottom[1]->shape(1)+j]);
			switch(value){
			  case 2:
				group.push_back(j);
				break;
			  case 3:
				group1.push_back(j);
				break;
			  case 4:
				group2.push_back(j);
				break;
			  case 5:
				group3.push_back(j);
				break;
			  default:
				LOG(FATAL) << "The value "<<value<<" can't be sorted in any condition.";
				break;
			}
		}
		for (int j=0; j<group.size() && j<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group[j]);
		}
		for (int j=0; j<group1.size() && (j+group.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group1[j]);
		}
		for (int j=0; j<group2.size() && (j+group.size()+group1.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group2[j]);
		}
		for (int j=0; j<group3.size() && (j+group.size()+group1.size()+group2.size())<num_output_; j++){
		  indices_.push_back(i);
		  indices_.push_back(group3[j]);
		}
	  }
	}
  }

  const Dtype* bottom_data_g = bottom[0]->cpu_data();
  Dtype* top_data_g = gather_output_.mutable_cpu_data();
  vector<int> bottom_shape = bottom[0]->shape();
  for (int m = 0; m < indices_.size()/indices_N_; ++m) {
	  const int top_offset = m * gather_nd_size_;
	  int bottom_offset = 0;
	  for (int n = 0; n < indices_N_; ++n) {
		  int indices_value = indices_[m*indices_N_ + n];
		  int params_idx = bottom_shape[n];
		  CHECK_LT(indices_value, params_idx) << "indices value does not index into param dimension: " << n;
		  bottom_offset += indices_[m*indices_N_ + n] * bottom[0]->count(n + 1);
	  }
	  caffe_copy(gather_nd_size_,
			  bottom_data_g + bottom_offset, top_data_g + top_offset);
  }

  int num_boxes_= group.size();
  const Dtype *bottom_data = bottom[2]->cpu_data();
  const Dtype *bottom_rois = gather_output_.cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  // Assume the data is in NHWC format
  int image_height_= bottom[2]->shape(1);
  int image_width_= bottom[2]->shape(2);
  Crop_And_Resize(bottom_data, bottom_rois, top_data, num_boxes_, image_height_, image_width_);

  num_boxes_= group1.size();
  const Dtype *bottom_data1 = bottom[3]->cpu_data();
  bottom_rois += group.size() * 4;
  top_data += group.size() * crop_height_ * crop_width_ * channels_;
  // Assume the data is in NHWC format
  image_height_= bottom[3]->shape(1);
  image_width_= bottom[3]->shape(2);
  Crop_And_Resize(bottom_data1, bottom_rois, top_data, num_boxes_, image_height_, image_width_);

  num_boxes_= group2.size();
  const Dtype *bottom_data2 = bottom[4]->cpu_data();
  bottom_rois += group1.size() * 4;
  top_data += group1.size() * crop_height_ * crop_width_ * channels_;
  // Assume the data is in NHWC format
  image_height_= bottom[4]->shape(1);
  image_width_= bottom[4]->shape(2);
  Crop_And_Resize(bottom_data2, bottom_rois, top_data, num_boxes_, image_height_, image_width_);

  num_boxes_= group3.size();
  const Dtype *bottom_data3 = bottom[5]->cpu_data();
  bottom_rois += group2.size() * 4;
  top_data += group2.size() * crop_height_ * crop_width_ * channels_;
  // Assume the data is in NHWC format
  image_height_= bottom[5]->shape(1);
  image_width_= bottom[5]->shape(2);
  Crop_And_Resize(bottom_data3, bottom_rois, top_data, num_boxes_, image_height_, image_width_);
}

INSTANTIATE_CLASS(Where4GatherndCropLayer);
REGISTER_LAYER_CLASS(Where4GatherndCrop);


}  // namespace caffe
