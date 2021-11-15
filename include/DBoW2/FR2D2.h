/**
 * File: FR2D2.h
 * Date: November 2021
 * Author: Afroz Alam (Modified from Dorian Galvez-Lopez, Joe Menke)
 * Description: functions for R2D2 descriptors
 *
 */

#ifndef __D_T_F_R2D2__
#define __D_T_F_R2D2__

#include <opencv2/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate R2D2 descriptors
class FR2D2: protected FClass
{
public:

  /// Descriptor type
  typedef cv::Mat TDescriptor; // CV_32F
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;
  /// Descriptor length
  static const int L = 128;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  /**
   * Calculates the (squared) distance between two descriptors
   * @param a
   * @param b
   * @return (squared) distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);

  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors NxL CV_8U matrix
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const cv::Mat &descriptors, cv::Mat &mat);

};

} // namespace DBoW2

#endif

