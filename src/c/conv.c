// Copied from https://github.com/BVLC/caffe
void im2col_cpu(const float *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                float *data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if ((unsigned) input_row >= (unsigned) height) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if ((unsigned) input_col < (unsigned) width) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void col2im_cpu(const float *data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                float *data_im)
{
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if ((unsigned) input_row >= (unsigned) height) {
                        data_col += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if ((unsigned) input_col < (unsigned) width) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

void max_pool_cpu_unbatched(
        const float *input,
        const int pad,
        const int xh,
        const int xw,
        const int yh,
        const int yw,
        const int ch,
        const int b,
        const int size,
        const int stride,
        float *output,
        float *indexes,
        const float float_min
)
{
    for (int c = 0; c < ch; ++c) {
        const int c_base = xh * (c + b * ch);
        for (int i = 0; i < yh; ++i) {
            const int i_base = yw * (i + yh * (c + b * ch));
            int h_start = i*stride - pad;
            const int h_end = h_start + size > xh? xh : h_start + size;
            h_start = h_start > 0? h_start : 0;
            for (int j = 0; j < yw; ++j) {
                float max = float_min;
                int max_i = 0; // default
                int w_start = j*stride - pad;
                const int w_end = w_start + size > xw? xw : w_start + size;
                w_start = w_start > 0? w_start : 0;
                // in a window
                for (int h = h_start; h < h_end; ++h) {
                    const int rows = xw * (h + c_base);
                    for (int w = w_start; w < w_end; ++w) {
                        const int index = w + rows;
                        const float val = input[index];
                        if (val > max) {
                            max_i = index;
                            max   = val;
                        }
                    }
                }
                int out_index = j + i_base;
                output[out_index] = max;
                indexes[out_index] = max_i;
            }
        }
    }
}

// TODO: Handle the case where multiple maximum values appear in each window
// by use of a binary mask. This also requires to fix max_pool_backward_cpu etc.
void max_pool_cpu(
        const float *input,
        const int pad,
        const int xh,
        const int xw,
        const int yh,
        const int yw,
        const int ch,
        const int batch,
        const int size,
        const int stride,
        float *output,
        float *indexes,
        const float float_min
)
{
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < ch; ++c) {
            const int c_base = xh * (c + b * ch);
            for (int i = 0; i < yh; ++i) {
                const int i_base = yw * (i + yh * (c + b * ch));
                int h_start = i*stride - pad;
                const int h_end = h_start + size > xh? xh : h_start + size;
                h_start = h_start > 0? h_start : 0;
                for (int j = 0; j < yw; ++j) {
                    float max = float_min;
                    int max_i = 0; // default
                    int w_start = j*stride - pad;
                    const int w_end = w_start + size > xw? xw : w_start + size;
                    w_start = w_start > 0? w_start : 0;
                    // in a window
                    for (int h = h_start; h < h_end; ++h) {
                        const int rows = xw * (h + c_base);
                        for (int w = w_start; w < w_end; ++w) {
                            const int index = w + rows;
                            const float val = input[index];
                            if (val > max) {
                                max_i = index;
                                max   = val;
                            }
                        }
                    }
                    int out_index = j + i_base;
                    output[out_index] = max;
                    indexes[out_index] = max_i;
                }
            }
        }
    }
}

void max_pool_grad_cpu(
        const float *gy,
        const int yh,
        const int yw,
        const int c,
        const int batch,
        float *gx,
        const float *argmax
)
{
    const int until = yh * yw * c * batch;
    for (int i = 0; i < until; ++i) {
        gx[(int) *(argmax++)] += *(gy++);
    }
}


void max_pool_grad_grad_cpu(
        const float *ggx,
        const int yh,
        const int yw,
        const int c,
        const int batch,
        float *ggy, // compute this
        const float *argmax
)
{
    const int until = yh * yw * c * batch;
    for (int i = 0; i < until; ++i) {
        *(ggy++) = ggx[(int) *(argmax++)];
    }
}
