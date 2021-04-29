#include "mex.h"
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <algorithm>

using namespace std;

inline uint32_t getAddressOffsetImage (const int32_t& u,const int32_t& v,const int32_t& height) {
  return u*height+v;
}

void backgroundInterpolation(float* D,bool* D_val,const int32_t* dims) {
  
  // maximum length of interpolation
  int32_t speckle_size = 2000;
  
  // grab image width and height
  int32_t width  = dims[1];
  int32_t height = dims[0];
  
  // declare loop variables
  int32_t count,addr,v_first,v_last,u_first,u_last;
  float   d1,d2,d_ipol;
  
  // 1. Row-wise:
  // for each row do
  for (int32_t v=0; v<height; v++) {
    
    // init counter
    count = 0;
    
    // for each element of the row do
    for (int32_t u=0; u<width; u++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,height);
      
      // if disparity valid
      if (*(D_val+addr)) {
        
        // check if speckle is small enough
        if (count>=1 && count<=speckle_size) {
          
          // first and last value for interpolation
          u_first = u-count;
          u_last  = u-1;
          
          // if value in range
          if (u_first>0 && u_last<width-1) {
            
            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u_first-1,v,height));
            d2 = *(D+getAddressOffsetImage(u_last+1,v,height));
            //if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            d_ipol = min(d1,d2);
            
            // set all values to d_ipol
            for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++) {
              *(D+getAddressOffsetImage(u_curr,v,height)) = d_ipol;
              *(D_val+getAddressOffsetImage(u_curr,v,height)) = 1;
            }
          }
          
        }
        
        // reset counter
        count = 0;
      
      // otherwise increment counter
      } else {
        count++;
      }
    }
    
    // extrapolate to the left
    for (int32_t u=0; u<width; u++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,height);
      
      // if disparity valid
      if (*(D_val+addr)) {
        for (int32_t u2=max(u-speckle_size,0); u2<u; u2++) {
          *(D+getAddressOffsetImage(u2,v,height)) = *(D+addr);
          *(D_val+getAddressOffsetImage(u2,v,height)) = 1;
        }
        break;
      }
    }
    
    // extrapolate to the right
    for (int32_t u=width-1; u>=0; u--) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,height);
      
      // if disparity valid
      if (*(D_val+addr)) {
        for (int32_t u2=u; u2<=min(u+speckle_size,width-1); u2++) {
          *(D+getAddressOffsetImage(u2,v,height)) = *(D+addr);
          *(D_val+getAddressOffsetImage(u2,v,height)) = 1;
        }
        break;
      }
    }    
  }
  
  
  // 1. Column-wise:
  // for each row do
  for (int32_t u=0; u<width; u++) {
    
    // extrapolate to the top
    for (int32_t v=0; v<height; v++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,height);
      
      // if disparity valid
      if (*(D_val+addr)) {
        for (int32_t v2=max(v-speckle_size,0); v2<v; v2++) {
          *(D+getAddressOffsetImage(u,v2,height)) = *(D+addr);
          *(D_val+getAddressOffsetImage(u,v2,height)) = 1;
        }
        break;
      }
    }
    
    // extrapolate to the bottom
    for (int32_t v=height-1; v>=0; v--) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,height);
      
      // if disparity valid
      if (*(D_val+addr)) {
        for (int32_t v2=v; v2<=min(v+speckle_size,height-1); v2++) {
          *(D+getAddressOffsetImage(u,v2,height)) = *(D+addr);
          *(D_val+getAddressOffsetImage(u,v2,height)) = 1;
        }
        break;
      }
    }
  }
}

void mexFunction (int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) {

    // check for proper number of arguments
    if(nrhs!=2) 
        mexErrMsgTxt("Two inputs required (D,D_val).");
    if(nlhs!=0) 
        mexErrMsgTxt("No output required.");
    
    // check for proper argument types and sizes
    if(!mxIsSingle(prhs[0]) || mxGetNumberOfDimensions(prhs[0])!=2)
        mexErrMsgTxt("Input D must be a float matrix.");
    if(!mxIsLogical(prhs[1]) || mxGetNumberOfDimensions(prhs[1])!=2)
        mexErrMsgTxt("Input D_val must be a logical matrix.");
   
    // create input pointers
    float*   D             =   (float*)mxGetPr(prhs[0]);
    bool*    D_val         =   (bool*) mxGetPr(prhs[1]);

    // do computation
    const int32_t *dims = mxGetDimensions(prhs[0]);
    backgroundInterpolation(D,D_val,dims);
}
