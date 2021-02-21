#pragma once
#include <Eigen/Core>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include "skel.h"


struct OpenposeDetection
{
	OpenposeDetection() { type = SkelType::SKEL_TYPE_NONE; }
	OpenposeDetection(const SkelType& _type);
	OpenposeDetection Mapping(const SkelType& tarType);
	std::vector<Eigen::Matrix3Xf> Associate(const int& jcntThresh = 5);

	SkelType type;
	std::vector<Eigen::Matrix3Xf> joints; // x, y, score(0,1)\
											这里的joints在Matrix3Xf里的序号就是上面的candid id？\
											如何知道joints的种类？是1-19中的哪一种关节呢？\
											先假装不区分关节种类，在jaCandididx之处也是全部joints的排序号。
	std::vector<Eigen::MatrixXf> pafs;    // @TODO 所以这里的parsing就是代码里的paf，是连接同一视角内 ‘几个头-几个脖子’ 的边，边存在的概率接近1
};

std::vector<OpenposeDetection> ParseDetections(const std::string& filename);
void SerializeDetections(const std::vector<OpenposeDetection>& detections, const std::string& filename);

