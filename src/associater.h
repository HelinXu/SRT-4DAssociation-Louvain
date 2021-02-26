#pragma once
#include "skel.h"
#include "camera.h"
#include "openpose.h"
#include <list>


class Associater
{
public:
	Associater(const SkelType& type, const std::map<std::string, Camera>& cams);
	void SetDetection(const int& view, const OpenposeDetection& detection) { assert(detection.type == m_type);  m_detections[view] = detection; }
	void SetDetection(const std::string& serialNumber, const OpenposeDetection& detection) { SetDetection(std::distance(m_cams.begin(), m_cams.find(serialNumber)), detection); }
	void SetSkels3dPrev(const std::map<int, Eigen::Matrix4Xf>& _skels3dPrev) { m_skels3dPrev = _skels3dPrev; }
	const std::map<int, Eigen::Matrix3Xf>& GetSkels2d() const { return m_skels2d; }
	void SetMaxEpiDist(const float& _maxEpiDist) { m_maxEpiDist = _maxEpiDist; }
	void SetMaxTempDist(const float& _maxTempDist) { m_maxTempDist = _maxTempDist; }
	void SetMinAsgnCnt(const int& _minAsgnCnt) { m_minAsgnCnt = _minAsgnCnt; }
	virtual void Associate(int FrameCount) = 0;

protected:
	float m_maxEpiDist = 0.2f;
	float m_maxTempDist = 0.5f;
	int m_minAsgnCnt = 5;
	bool m_normalizeEdges = true;

	SkelType m_type;
	std::map<std::string, Camera> m_cams;
	std::vector<OpenposeDetection> m_detections;  // @TODO

	std::map<int, Eigen::Matrix4Xf> m_skels3dPrev;
	std::map<int, Eigen::Matrix3Xf> m_skels2d;

	std::vector<std::vector<Eigen::VectorXi>> m_assignMap;
	std::map<int, Eigen::MatrixXi> m_personsMap;
	std::vector<std::vector<Eigen::Matrix3Xf>> m_jointRays;
	std::vector<std::vector<std::vector<Eigen::MatrixXf>>> m_epiEdges;	// m_epiEdge[jIdx][viewA][viewB](jaCandiIdx, jbCandiIdx) @TODO\
																		epiEdge是连接不同视角的同一种关节的边\
																		这个矩阵在jaCandiIdx和jb的交点那个单元格上存的值\
																		是该条边已经计算得出的置信度，是个（0，1）的值\
																		能不能再讲一下这个jaCandiIdx是什么\

	std::vector<std::vector<Eigen::MatrixXf>> m_tempEdges;				// m_tempEdge[jIdx][view](pIdx, jCandiIdx) @TODO
	int m_FrameCount;
	
	void Initialize();
	void CalcJointRays();
	void CalcPafEdges();
	void CalcEpiEdges();
	void CalcTempEdges();
	void CalcSkels2d();
	float Point2LineDist(const Eigen::Vector3f& pA, const Eigen::Vector3f& pB, const Eigen::Vector3f& ray);
	float Line2LineDist(const Eigen::Vector3f& pA, const Eigen::Vector3f& rayA, const Eigen::Vector3f& pB, const Eigen::Vector3f& rayB);
};
