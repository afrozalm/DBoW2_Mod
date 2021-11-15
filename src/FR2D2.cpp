/**
 * File: FR2D2.cpp
 * Date: November 2021
 * Author: Afroz Alam (Modified from Dorian Galvez-Lopez, Joe Menke)
 * Description: functions for R2D2 descriptors
 * License: see the LICENSE.txt file
 *
 */

#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <limits.h>

#include "FR2D2.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FR2D2::meanValue(const std::vector<FR2D2::pDescriptor> &descriptors,
  FR2D2::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    mean.release();
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    float s = descriptors.size();
    mean = cv::Mat::zeros(1, FR2D2::L, CV_32F);
    float *p = mean.ptr<float>();

    vector<FR2D2::pDescriptor>::const_iterator it;
    for(it = descriptors.begin(); it != descriptors.end(); ++it)
    {
      const FR2D2::TDescriptor &desc = **it;
      const float *q = desc.ptr<float>();
      for(int i = 0; i < FR2D2::L; i += 4)
      {
        p[i  ] += q[i  ] / s;
        p[i+1] += q[i+1] / s;
        p[i+2] += q[i+2] / s;
        p[i+3] += q[i+3] / s;
      }
    }
  }
}

// --------------------------------------------------------------------------

double FR2D2::distance(const FR2D2::TDescriptor &a,
  const FR2D2::TDescriptor &b)
{
  double sqd = 0.;
  const float *p = a.ptr<float>(), *q = b.ptr<float>();
  for(int i = 0; i < FR2D2::L; i += 4)
  {
    sqd += (p[i  ] - q[i  ])*(p[i  ] - q[i  ]);
    sqd += (p[i+1] - q[i+1])*(p[i+1] - q[i+1]);
    sqd += (p[i+2] - q[i+2])*(p[i+2] - q[i+2]);
    sqd += (p[i+3] - q[i+3])*(p[i+3] - q[i+3]);
  }
  return sqd;
}

// --------------------------------------------------------------------------

std::string FR2D2::toString(const FR2D2::TDescriptor &a)
{
  stringstream ss;
  const float *p = a.ptr<float>();
  for(int i = 0; i < FR2D2::L; ++i)
  {
    ss << p[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------

void FR2D2::fromString(FR2D2::TDescriptor &a, const std::string &s)
{
  a.create(1, FR2D2::L, CV_32F);
  float *p = a.ptr<float>();

  stringstream ss(s);
  for(int i = 0; i < FR2D2::L; ++i)
  {
    ss >> p[i];
  }
}

// --------------------------------------------------------------------------

void FR2D2::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const size_t N = descriptors.size();
  const size_t L = FR2D2::L;

  mat.create(N, L, CV_32F);
  float *p = mat.ptr<float>();

  for(size_t i = 0; i < N; ++i)
  {
    const float *desc = descriptors[i].ptr<float>();
    float *p = mat.ptr<float>(i);

    for(int j = 0; j < L; ++j, ++p)
    {
      *p = desc[j];
    }
  }
}

// --------------------------------------------------------------------------

void FR2D2::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{
  descriptors.convertTo(mat, CV_32F);
  return;
}

// --------------------------------------------------------------------------

} // namespace DBoW2

