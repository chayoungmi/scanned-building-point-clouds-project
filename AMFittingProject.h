#pragma once
//#include "AMPointCloud2Wnd.h"
//#include "AMPtLib.h"

#include <vector>
#include "Common/AMMath.h"

#include "AMPoint2.h"
#include "AMPrimitive.h"

class AMPointProject2;

using std::vector;

#define MAX_REQUIRED_POINT_NUM		20


class AMFittingProject
{
public:
	AMFittingProject(void);
	~AMFittingProject(void);

	static		AMFittingProject* GetFittingProject() { return &m_fittingProject; }
	AMPointProject2*	GetPointProject() { return m_pPointProject; }
	void SetPointProject( AMPointProject2* pPointPrj) { m_pPointProject = pPointPrj; }

	void Initialize();
	void Render( );

	// ���� �� �ʿ��� distance, angle parameter �ڵ� ���� ���
	float	GetAutoDistanceThreshold( vector<BMVector3f>* pPointArray, int sampleNum, vector<int>* pPointIdxList = nullptr );
	float	GetAutoAngleThreshold();

	
	// RANSAC�� �ʿ��� parameter ���� �Է����� �޴� �Լ���
	PlanePrimitive	FitPlane( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, bool isAxisAligned, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	LinePrimitive FitLine( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	CirclePrimitive3D FitCircle3D( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	SpherePrimitive	FitSphere( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray,
		int maxItr,  float numCloseDataRatio, float inlierRatio, float distThreshold, float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	CylinderPrimitive	FitCylinder( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray,
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	ConePrimitive	FitCone( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	TorusPrimitive FitTorus( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	// �ּ����� parameter�� �Է����� �޴� �Լ���
	PlanePrimitive	FitPlane( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold,  bool isAxisAligned, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	LinePrimitive FitLine( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	CirclePrimitive3D FitCircle3D( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	SpherePrimitive	FitSphere( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	CylinderPrimitive	FitCylinder( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr,
		CylinderPrimitive* pOutRANSACCylinder = nullptr, CylinderPrimitive* pOutLMOptCylinder = nullptr );
	
	ConePrimitive	FitCone( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );

	TorusPrimitive FitTorus( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold , 
		vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList = nullptr );
	
	//////////////////////////////////////////////////////////////////////////
	// multi primitive
	// FitPlane�� ������, ������ ���� ��ŭ fitting�� �õ��ϰ�, ������ ������ �����Ѵ�. fitting�� plane���� pOutPlanes vector�� �����Ѵ�.
	// Fitting�� plane���� new�� �����Ǿ� ����Ʈ�� ����ȴ�.
	// threshold�� �ڵ����� ����Ϸ��� ���� 0���� �����Ѵ�. distThreshold, angleThreshold�� ���� ����ȴ�.
	int	FitMultiPlane( std::vector<PlanePrimitive*>* pOutPlanePtrs, 
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray, 
		float distThreshold, 	float angleThreshold, int maxPrimitives, bool isAxisAligned, float stopRatio = 1.0f );

	int FitMultiLine( std::vector<LinePrimitive*>* pOutLines,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	int FitMultiLineOnXYPlane( std::vector<LinePrimitive*>* pOutLines,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	int FitMultiCircle3D( std::vector<CirclePrimitive3D*>* pOutCircles,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	int FitMultiSphere( std::vector<SpherePrimitive*>* pOutSpheres,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	int FitMultiCylinder( std::vector<CylinderPrimitive*>* pOutCylinders,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f,
		std::vector<CylinderPrimitive*>* pOutRANSACCylinders = nullptr, std::vector<CylinderPrimitive*>* pOutLMOptCylinders = nullptr );

	int FitMultiCone( std::vector<ConePrimitive*>* pOutCones,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	int FitMultiTorus( std::vector<TorusPrimitive*>* pOutTori,
		std::vector<BMVector3f>* pPointArray, std::vector<BMVector3f>* pNormalArray,
		float distThreshold, float angleThreshold, int maxPrimitives, float stopRatio = 1.0f );

	// ��� ������ �Ͽ� ����
	void RegisterCallbackFunctionForRendering( void(*callback)(void* pObj, IPrimitive*, CString&, CString& )); 
	void SavePrimitiveRenderResult( int bestInlierNum, IPrimitive* pPrimitive );
	
	// global setting
	static bool	GetVerbose() { return m_bVerbose; }
	static void	SetVerbose( bool bVerbose ) { m_bVerbose = bVerbose; }
	static bool	IsStaticRandomSequence() { return m_bStaticRandomSequence; }
	static void SetStaticRandomSequence( bool bStatic ) { m_bStaticRandomSequence = bStatic; }
	
public:
	void(*_callback)( void* pObj, IPrimitive*, CString& fileName, CString& description );

protected:
	bool GetMainAxesAlignedSpecificAxis( vector<BMVector3f>* pPointArray, vector<int>* pInlierIdxList, 
		BMVector3f specificAxis, PlanePrimitive* pPlane );


protected:
	static AMFittingProject		m_fittingProject;
	AMPointProject2*			m_pPointProject;

	int									m_renderImgCnt;

	static bool	m_bVerbose;
	static bool m_bStaticRandomSequence;

public:
	
};

inline AMFittingProject*	AMGetFittingProject() {  return AMFittingProject::GetFittingProject(); }

