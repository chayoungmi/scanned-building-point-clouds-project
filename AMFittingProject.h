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

	// 피팅 시 필요한 distance, angle parameter 자동 설정 기능
	float	GetAutoDistanceThreshold( vector<BMVector3f>* pPointArray, int sampleNum, vector<int>* pPointIdxList = nullptr );
	float	GetAutoAngleThreshold();

	
	// RANSAC에 필요한 parameter 전부 입력으로 받는 함수들
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

	// 최소한의 parameter만 입력으로 받는 함수들
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
	// FitPlane과 같지만, 지정한 개수 만큼 fitting을 시도하고, 성공한 개수를 리턴한다. fitting된 plane들은 pOutPlanes vector에 저장한다.
	// Fitting된 plane들은 new로 생성되어 리스트에 저장된다.
	// threshold를 자동으로 계산하려면 값을 0으로 설정한다. distThreshold, angleThreshold에 각각 적용된다.
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

	// 결과 렌더링 하여 저장
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

