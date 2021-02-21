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
											�����joints��Matrix3Xf�����ž��������candid id��\
											���֪��joints�����ࣿ��1-19�е���һ�ֹؽ��أ�\
											�ȼ�װ�����ֹؽ����࣬��jaCandididx֮��Ҳ��ȫ��joints������š�
	std::vector<Eigen::MatrixXf> pafs;    // @TODO ���������parsing���Ǵ������paf��������ͬһ�ӽ��� ������ͷ-�������ӡ� �ıߣ��ߴ��ڵĸ��ʽӽ�1
};

std::vector<OpenposeDetection> ParseDetections(const std::string& filename);
void SerializeDetections(const std::vector<OpenposeDetection>& detections, const std::string& filename);

