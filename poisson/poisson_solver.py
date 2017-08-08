import cv2
import numpy as np

# super parameters
LOOP_MAX = 1000
EPS = 2.2204e-016
NUM_NEIGHBOR = 4

def poisson_solver(img_src, img_dst, img_mask, offset):
	result = np.zeros(img_dst.shape, img_dst.dtype);
  for(int channel=0; channel<3; channel++) {
    int i, j, loop, neighbor, count_neighbors, flag_edge, ok;
    float error, sum_f, sum_fstar, sum_vpq, fp, fq, gp, gq;
    int naddr[NUM_NEIGHBOR][2] = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    Mat img_new = (cv::Mat_<double>(img_dst.rows,img_dst.cols));
    for(i=0; i<img_dst.rows; i++){
      for(j=0; j<img_dst.cols; j++){
        img_new.ptr<double>(i)[j] = (double)img_dst.ptr<Vec3b>(i)[j][channel];
      }
    }
    for(loop=0; loop<LOOP_MAX; loop++){
      ok = 1;
      for(i=0; i<img_mask.rows; i++){
        for(j=0; j<img_mask.cols; j++){
          if((int)img_mask.ptr<Vec3b>(i)[j][0] > 0){
            sum_f = 0.0;
            sum_fstar = 0.0;
            sum_vpq = 0.0;
            count_neighbors = 0;
            flag_edge = 0;
            for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++){
              if((int)img_mask.ptr<Vec3b>(i+naddr[neighbor][0])[j+naddr[neighbor][1]][0] == 0){
                flag_edge = 1;
                break;
              }
            }
            if(flag_edge == 0) {
              for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++) {
                if(i+offset[0]+naddr[neighbor][0] >= 0
                   && j+offset[1]+naddr[neighbor][1] >= 0
                   && i+offset[0]+naddr[neighbor][0] < img_dst.rows
                   && j+offset[1]+naddr[neighbor][1] < img_dst.cols){
                  sum_f += img_new.ptr<double>(i+offset[0]+naddr[neighbor][0])[j+offset[1]+naddr[neighbor][1]];
                  sum_vpq += (float) img_src.ptr<Vec3b>(i)[j][channel]
                    - (float) img_src.ptr<Vec3b>(i+naddr[neighbor][0])[j+naddr[neighbor][1]][channel];
                  count_neighbors++;
                }
              }  
            } else {
              for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++) {
                if(i+offset[0]+naddr[neighbor][0] >= 0
                   && j+offset[1]+naddr[neighbor][1] >= 0
                   && i+offset[0]+naddr[neighbor][0] < img_dst.rows
                   && j+offset[1]+naddr[neighbor][1] < img_dst.cols){
                  fp = (float) img_dst.ptr<Vec3b>(i+offset[0])[j+offset[1]][channel];
                  fq = (float) img_dst.ptr<Vec3b>(i+offset[0]+naddr[neighbor][0])[j+offset[1]+naddr[neighbor][1]][channel];
                  gp = (float) img_src.ptr<Vec3b>(i)[j][channel];
                  gq = (float) img_src.ptr<Vec3b>(i+naddr[neighbor][1])[j+naddr[neighbor][1]][channel];
                  sum_fstar += fq;
                  if(fabs(fp - fq) > fabs(gp - gq)) {
                    sum_vpq += fp - fq;
                  } else {
                    sum_vpq += gp - gq;
                  }
                  count_neighbors++;
                }
              }
            }
            fp = (sum_f + sum_fstar + sum_vpq) / (float)count_neighbors;
            error = fabs(fp - img_new.ptr<double>(i+offset[0])[j+offset[1]]);
            if(ok && error > EPS * (1+fabs(fp))) {
              ok = 0;
            }
            img_new.ptr<double>(i+offset[0])[j+offset[1]] = fp;
          }
        }
      }
      if(ok){
        break;
      }
    }
    for(i=0; i<img_dst.rows; i++){
      for(j=0; j<img_dst.cols; j++){
        if(img_new.ptr<double>(i)[j] > 255){
          img_new.ptr<double>(i)[j] = 255.0;
        }
        else if(img_new.ptr<double>(i)[j] < 0){
          img_new.ptr<double>(i)[j] = 0.0;
        }
        result.ptr<Vec3b>(i)[j][channel] = (uchar)img_new.ptr<double>(i)[j];
      }
    }
  }
  return result;