#include "StdAfx.h"
#include "AMFittingProject.h"

#include "AMPointProject2.h"
#include "AMPointCloudRANSAC.h"
#include "AMPrimitive.h"
#include "AMLeastSquareFitting.h"
#include "AMPtMath.h"


AMFittingProject AMFittingProject::m_fittingProject;
bool			AMFittingProject::m_bVerbose = false;
bool			AMFittingProject::m_bStaticRandomSequence = false;

AMFittingProject::AMFittingProject(void)
{
	m_renderImgCnt = 0;
	m_pPointProject = nullptr;
}

AMFittingProject::~AMFittingProject(void)
{
}

void AMFittingProject::Initialize()
{
	m_renderImgCnt = 0;
}

void AMFittingProject::RegisterCallbackFunctionForRendering( void(*callback)(void* pObj, IPrimitive*, CString&, CString&) )
{
	_callback = callback;
}

void AMFittingProject::SavePrimitiveRenderResult( int bestInlierNum, IPrimitive* pPrimitive)
{
	if( !m_pPointProject ) 
		return ;

	if ( !_callback ) return;

	CString errorInfo, fileName;
	std::string paramInfo = "";

	//callback ( draw Frame to Image ) 
	char tmpChar[100];
	sprintf_s( tmpChar, 100, "Primitive_%05d.png", m_renderImgCnt );
	fileName = tmpChar;
	errorInfo = _T("");
	paramInfo = pPrimitive->GetParameterInfo();
	sprintf_s( tmpChar, 100, "Inlier Point Num : %d\n", bestInlierNum );
	errorInfo.Append( tmpChar );
	errorInfo.Append( paramInfo.c_str() );
	_callback( GetPointProject(), pPrimitive, fileName, errorInfo );
	
	m_renderImgCnt++;
}

float	AMFittingProject::GetAutoDistanceThreshold( vector<BMVector3f>* pPointArray, int sampleNum,  vector<int>* pPointIdxList )
{
	//dlg parameter 자동 설정
	if( sampleNum <= 0 )
		return 0.0f;

	float averageLength = AMPtMath::GetAverageNearestNeighborDistance( pPointArray, sampleNum, pPointIdxList );
	averageLength = averageLength * 1.5f; // 여유분
	float distance = AMRoundOffScale( averageLength, 0.01f, 10.0f ) ;
	return distance;	
	return averageLength;
}

float	AMFittingProject::GetAutoAngleThreshold()
{
	return 0.2f;
}

bool AMFittingProject::GetMainAxesAlignedSpecificAxis( vector<BMVector3f>* pPointArray, vector<int>* pInlierIdxList, 
																								BMVector3f specificAxis, PlanePrimitive* pPlane )
{
	// mean value
	BMVector3d meanValue = AMPtMath::GetAverageValue( pPointArray, pInlierIdxList );
	pPlane->m_center = (BMVector3f)meanValue;

	// 축 구하고
	BMVector3f virtualZAxis;
	virtualZAxis = pPlane->m_normal.CrossProduct( specificAxis );
	virtualZAxis.Normalize();

	//AMMathUtil::GetProjectedPointOnPlane( specificAxis, pPlane->m_normal, pPlane->m_center, virtualZAxis );
	//virtualZAxis = virtualZAxis - pPlane->m_center ;

	BMVector3f mainAxis = pPlane->m_normal.CrossProduct( virtualZAxis );
	mainAxis.Normalize();

	if( !pPlane->m_planeAxes.empty() )
		pPlane->m_planeAxes.clear();

	pPlane->m_planeAxes.push_back( virtualZAxis );
	pPlane->m_planeAxes.push_back( mainAxis );

	//
	bool isSucceed = AMPtMath::GetRectangleRangeVertices3DAlignedAxes( pPointArray, pInlierIdxList, pPlane->m_normal,
									&(pPlane->m_planeAxes), pPlane->m_center, &(pPlane->m_vertices) );
	return isSucceed;
}

//------------------------------------------------------------------------------------------------------------------------------
// Primitive Fitting 시 RANSAC에 필요한 모든 parameter를 입력으로 받는 함수

PlanePrimitive AMFittingProject::FitPlane( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
									int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, bool isAxisAligned, 
									vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	PlanePrimitive modelParam;

	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMPlaneRANSAC<PlanePrimitive> ransac;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	printf("[RANSAC Parameter] : maxItrNum : %d, distThreshold : %f, angleThreshold : %f .\n", 
		ransac.m_maxItrNum, ransac.m_thresholdDistance, ransac.m_thresholdAngle );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList,  &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	// LM
	AMLeastSquareFitting<PlanePrimitive> LM;
	PlanePrimitive finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose )
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	// plane의 boundary edge
	if( finalModel.IsValidate() )
	{
		if( isAxisAligned )
		{
			BMVector3f zAxis(0, 0, 1 );
			bool isSucceed = GetMainAxesAlignedSpecificAxis( pPointArray, pOutInlierIdxList, zAxis, &finalModel );
			if( !isSucceed )
				printf(" plane edge 생성 실패. \n");
		}
		else
		{
			bool isSucceed = AMPtMath::EstimatePlaneRectangleVertices3D( pPointArray, pOutInlierIdxList, 
				finalModel.m_normal, &finalModel.m_vertices, &finalModel.m_planeAxes, &finalModel.m_center );
			if( !isSucceed )
				printf(" plane edge 생성 실패. \n");
		}
	}
	float timeBoundaryEdge = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	AMLog( "Plane fitting: total %.3f sec (RANSAC %.3f, refining %.3f, boundaryEdge=  %.3f)", timeTotal, timeRansac, timeRefinement, timeBoundaryEdge );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Plane fitting: total %.3f sec (RANSAC %.3f, refining %.3f, boundaryEdge=  %.3f)\n", timeTotal, timeRansac, timeRefinement, timeBoundaryEdge );

	return finalModel;
}

LinePrimitive AMFittingProject::FitLine( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	LinePrimitive modelParam;

	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMLineRANSAC<LinePrimitive> ransac;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	// LM
	AMLeastSquareFitting<LinePrimitive> LM;
	LinePrimitive finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}
	
	// line의 boundary edge
	if( finalModel.IsValidate() )
	{
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, 
			finalModel.m_direction, finalModel.m_pointOnLine, finalModel.m_lineSegment.m_start, finalModel.m_lineSegment.m_end );
		if( !isSucceed )
			printf(" plane edge 생성 실패. \n");
	}
	float timeAxisBoundary = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	AMLog( "Line fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Line fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)\n", timeTotal, timeRansac, timeRefinement,timeAxisBoundary );

	return finalModel;
}

CirclePrimitive3D AMFittingProject::FitCircle3D( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	CirclePrimitive3D modelParam;
	PlanePrimitive tempPlane;

	AMStopWatch stopWatch;
	float timePlaneFit = -1, timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();

	//1. plane 우선 찾는다.
	tempPlane = FitPlane( pPointArray, pNormalArray, maxItr, numCloseDataRatio, inlierRatio, distThreshold, angleThreshold, false,
		pOutInlierIdxList, pOutOutlierIdxList, pPointIdxList );
	timePlaneFit = stopWatch.LapSecond();
	
	//2. plane 상의 circle 찾는다.
	AMCircle3DRANSAC<CirclePrimitive3D> ransac;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );
	ransac.SetInitialPlaneData( &tempPlane, &modelParam );
	
	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}
		
	// LM
	AMLeastSquareFitting<CirclePrimitive3D> LM;
	CirclePrimitive3D finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	AMLog( "Circle3D fitting: total %.3f sec (Plane fitting %.3f, RANSAC %.3f, refining %.3f)", timeTotal, timePlaneFit, timeRansac, timeRefinement);
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Circle3D fitting: total %.3f sec (Plane fitting %.3f, RANSAC %.3f, refining %.3f)\n", timeTotal, timePlaneFit, timeRansac, timeRefinement);

	return finalModel;
}

SpherePrimitive AMFittingProject::FitSphere( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	SpherePrimitive modelParam;
	
	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();
	
	AMSphereRANSAC<SpherePrimitive> ransac;
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();
	
	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	//LM
	AMLeastSquareFitting<SpherePrimitive> LM;

	SpherePrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	AMLog( "Sphere fitting: total %.3f sec (RANSAC %.3f, refining %.3f)", timeTotal, timeRansac, timeRefinement );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Sphere fitting: total %.3f sec (RANSAC %.3f, refining %.3f)\n", timeTotal, timeRansac, timeRefinement );

	return finalModel;
}


CylinderPrimitive AMFittingProject::FitCylinder( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold,
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	CylinderPrimitive modelParam;

	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeAxisBoundary = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMCylinderRANSAC<CylinderPrimitive> ransac;
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );

	if( !pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}
	timeRansac = stopWatch.LapSecond();

	//LM
	AMLeastSquareFitting<CylinderPrimitive> LM;

	CylinderPrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}
	timeRefinement = stopWatch.LapSecond();

	// cylinder의 boundary ( 축만 알면 됨 )
	if( finalModel.IsValidate() )
	{
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, finalModel.m_axis, finalModel.m_center, 
			finalModel.m_bottomOnAxis, finalModel.m_topOnAxis ) ;

		if( !isSucceed )
			printf(" cylinder edge 생성 실패. \n");
	}
	timeAxisBoundary = stopWatch.LapSecond();
	stopWatch.Stop();
	timeTotal = stopWatch.ReadResultSecond();

	AMLog( "Cylinder fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Cylinder fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)\n", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );

	return finalModel;
}

ConePrimitive AMFittingProject::FitCone( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	ConePrimitive modelParam;

	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMConeRANSAC<ConePrimitive> ransac;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}
	
	//LM
	AMLeastSquareFitting<ConePrimitive> LM;

	ConePrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	// cone의 boundary ( 축만 알면 됨 )
	if( finalModel.IsValidate() )
	{
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, finalModel.m_axis, finalModel.m_apex, 
			finalModel.m_bottomOnAxis, finalModel.m_topOnAxis ) ;

		if( !isSucceed )
			printf(" cone edge 생성 실패. \n");
	}
	float timeAxisBoundary = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	AMLog( "Cone fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary=  %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	printf( "Cone fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary=  %.3f)\n", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );

	return finalModel;
	
}

TorusPrimitive AMFittingProject::FitTorus( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	int maxItr, float numCloseDataRatio, float inlierRatio, float distThreshold, 	float angleThreshold,
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	TorusPrimitive modelParam;

	if( pPointIdxList != nullptr )
	{
		if ( pPointIdxList->size() < MAX_REQUIRED_POINT_NUM )
		{
			//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
			return modelParam;
		}
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMTorusRANSAC<TorusPrimitive> ransac;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameter( maxItr, inlierRatio, numCloseDataRatio );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	//LM
	AMLeastSquareFitting<TorusPrimitive> LM;

	TorusPrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	timeTotal = stopWatch.StopSecond();

	AMLog( "Torus fitting: total %.3f sec (RANSAC %.3f, refining %.3f)", timeTotal, timeRansac, timeRefinement );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Torus fitting: total %.3f sec (RANSAC %.3f, refining %.3f)\n", timeTotal, timeRansac, timeRefinement );

	return finalModel;
}

//------------------------------------------------------------------------------------------------------------------------------
// Primitive Fitting 시 최소한의 parameter를 입력으로 하고, 나머지는 내부에서 자동 계산.


PlanePrimitive AMFittingProject::FitPlane( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold, bool isAxisAligned, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	PlanePrimitive modelParam;
	int totalInputPointNum = 0;

	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMPlaneRANSAC<PlanePrimitive> ransac;

	// RANSAC에 필요한 parameter 자동 계산
	// maxiteration 
	size_t numCloseData = pPointArray->size();
	
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList,  &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
		printf("Inlier Points Num : %d\n", pOutInlierIdxList->size() );
	}

	if( pOutInlierIdxList->empty() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	// LM
	AMLeastSquareFitting<PlanePrimitive> LM;
	PlanePrimitive finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose )
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	// plane의 boundary edge
	if( finalModel.IsValidate() )
	{
		if( isAxisAligned )
		{
			BMVector3f zAxis(0, 0, 1 );
			bool isSucceed = GetMainAxesAlignedSpecificAxis( pPointArray, pOutInlierIdxList, zAxis, &finalModel );
			if( !isSucceed )
				printf(" plane edge 생성 실패. \n");
		}
		else
		{
			bool isSucceed = AMPtMath::EstimatePlaneRectangleVertices3D( pPointArray, pOutInlierIdxList, 
				finalModel.m_normal, &finalModel.m_vertices, &finalModel.m_planeAxes, &finalModel.m_center );
			if( !isSucceed )
				printf(" plane edge 생성 실패. \n");
		}

		// plane normal과 points의 normal 값을 비교하여 plane 아닌 경우를 제외시키도록 한다.
		bool bPlaneDescribedWell = AMPtMath::IsDescribedPlane( finalModel.m_normal, pPointArray, pNormalArray, pOutInlierIdxList );
		if( !bPlaneDescribedWell )
		{
			finalModel.m_normal.SetZero();
			pOutInlierIdxList->clear();
			pOutOutlierIdxList->clear();
			//printf(" plane data가 아니므로 피팅 실패. \n");
		}

	}
	float timeBoundaryEdge = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	pOutInlierIdxList->clear();
	pOutOutlierIdxList->clear();
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Plane fitting: total %.3f sec (RANSAC %.3f, refining %.3f, boundaryEdge=  %.3f)", timeTotal, timeRansac, timeRefinement, timeBoundaryEdge );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Plane fitting: total %.3f sec (RANSAC %.3f, refining %.3f, boundaryEdge=  %.3f)\n", timeTotal, timeRansac, timeRefinement, timeBoundaryEdge );
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum, pOutInlierIdxList->size() );

	return finalModel;
}

LinePrimitive AMFittingProject::FitLine( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	LinePrimitive modelParam;
	int totalInputPointNum = 0;

	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMLineRANSAC<LinePrimitive> ransac;

	size_t numCloseData = pPointArray->size();
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	// LM
	AMLeastSquareFitting<LinePrimitive> LM;
	LinePrimitive finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	// line의 boundary edge
	if( finalModel.IsValidate() )
	{
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, 
			finalModel.m_direction, finalModel.m_pointOnLine, finalModel.m_lineSegment.m_start, finalModel.m_lineSegment.m_end );
		if( !isSucceed )
			printf(" plane edge 생성 실패. \n");
	}
	float timeAxisBoundary = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	pOutInlierIdxList->clear();
	pOutOutlierIdxList->clear();
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Line fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Line fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)\n", timeTotal, timeRansac, timeRefinement,timeAxisBoundary );
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum, pOutInlierIdxList->size() );

	return finalModel;
}

CirclePrimitive3D AMFittingProject::FitCircle3D( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	CirclePrimitive3D modelParam;
	PlanePrimitive tempPlane;

	AMStopWatch stopWatch;
	float timePlaneFit = -1, timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();

	//1. plane 우선 찾는다.
	vector<int> planeInlierIdxList, planeInOutlierdxList;
	tempPlane = FitPlane( pPointArray, pNormalArray, distThreshold, angleThreshold, false,
		&planeInlierIdxList, &planeInOutlierdxList, pPointIdxList );
	timePlaneFit = stopWatch.LapSecond();
	 
	//2. plane 상의 circle 찾는다.
	
	AMCircle3DRANSAC<CirclePrimitive3D> ransac;
	size_t numCloseData = pPointArray->size();

	// plane 상의 point list들로 이루어진 circle 구한다.
	
	// input point data를 plane data로 교체한다. 
	if( !planeInlierIdxList.empty() )
		pPointIdxList = &planeInlierIdxList ;

	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );
	ransac.SetInitialPlaneData( &tempPlane, &modelParam );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose )
	{
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	// LM
	AMLeastSquareFitting<CirclePrimitive3D> LM;
	CirclePrimitive3D finalModel;

	bool isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	if( !isSuccess )
		printf(" LM 실패\n");

	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	pOutInlierIdxList->clear();
	pOutOutlierIdxList->clear();
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Circle3D fitting: total %.3f sec (Plane fitting %.3f, RANSAC %.3f, refining %.3f)", timeTotal, timePlaneFit, timeRansac, timeRefinement);
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Circle3D fitting: total %.3f sec (Plane fitting %.3f, RANSAC %.3f, refining %.3f)\n", timeTotal, timePlaneFit, timeRansac, timeRefinement);
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", pPointIdxList ? pPointIdxList->size() : pPointArray->size(), pOutInlierIdxList->size() );
	
	// 최종 outlier 또한 plane 이외의 점으로 제한하도록 한다.
	if( !planeInOutlierdxList.empty() )
		pOutOutlierIdxList->swap( planeInOutlierdxList );

	return finalModel;
}


SpherePrimitive AMFittingProject::FitSphere( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	SpherePrimitive modelParam;
	int totalInputPointNum = 0;

	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMSphereRANSAC<SpherePrimitive> ransac;
	size_t numCloseData = pPointArray->size();
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	//LM
	AMLeastSquareFitting<SpherePrimitive> LM;

	SpherePrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Sphere fitting: total %.3f sec (RANSAC %.3f, refining %.3f)", timeTotal, timeRansac, timeRefinement );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Sphere fitting: total %.3f sec (RANSAC %.3f, refining %.3f)\n", timeTotal, timeRansac, timeRefinement );
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum , pOutInlierIdxList->size() );

	return finalModel;
}

CylinderPrimitive AMFittingProject::FitCylinder( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold,
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList,
	CylinderPrimitive* pOutRANSACCylinder, CylinderPrimitive* pOutLMOptCylinder )
{
	CylinderPrimitive modelParam;
	int totalInputPointNum = 0;
	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}

	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeAxisBoundary = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMCylinderRANSAC<CylinderPrimitive> ransac;
	size_t numCloseData = pPointArray->size();
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		modelParam.m_radius = 0.0f; // set to invalid model
	}

	if( !modelParam.IsValidate() )
	{
		pOutInlierIdxList->clear();
		pOutOutlierIdxList->clear();
		return modelParam;
	}

	if ( pOutRANSACCylinder )
	{
		AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, modelParam.m_axis, modelParam.m_center,
			modelParam.m_bottomOnAxis, modelParam.m_topOnAxis );
		*pOutRANSACCylinder = modelParam;
		pOutRANSACCylinder->SetAlgType( AM_RANSAC_TYPE );
		pOutRANSACCylinder->Normalize();
	}

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}
	timeRansac = stopWatch.LapSecond();
	
	//// RANSAC 결과 파일로 저장
	//if ( m_bVerbose ) 
	//{
	//	FILE* pParameterFile;
	//	FILE* pPointFile;
	//	pParameterFile = fopen( "Cylinder_parameter.txt", "w");
	//	pPointFile = fopen( "Cylinder_point.txt", "w");
	//	// ransac 결과 parameters
	//	fprintf( pParameterFile, "%f %f %f\n",  modelParam.m_axis.x, modelParam.m_axis.y, modelParam.m_axis.z );
	//	fprintf( pParameterFile, "%f %f %f\n",  modelParam.m_center.x, modelParam.m_center.y, modelParam.m_center.z );
	//	fprintf( pParameterFile, "%f \n",  modelParam.m_radius );
	//	
	//	// ransac 결과 points 저장
	//	fprintf( pPointFile, "%d\n",  pOutInlierIdxList->size() );
	//	int index = 0;
	//	BMVector3f outPoint; 
	//	for( int i = 0; i < pOutInlierIdxList->size(); i++)
	//	{
	//		index = pOutInlierIdxList->at(i) ;
	//		outPoint = pPointArray->at( index );
	//		fprintf( pPointFile, "%f %f %f\n",  outPoint.x, outPoint.y, outPoint.z );
	//	}
	//	fclose( pParameterFile );
	//	fclose( pPointFile );
	//}

	// lm openSource test
	CylinderPrimitive finalModel = modelParam;
	if( pOutLMOptCylinder )
	{
		AMLMOptimization<CylinderPrimitive> LMOpt;
		LMOpt.Initialize( pPointArray, pOutInlierIdxList, finalModel );
		LMOpt.Run_LMOptimize();
		finalModel = LMOpt.m_model;

		// save LM result
		if( finalModel.IsValidate() )
		{
			bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, finalModel.m_axis, finalModel.m_center, 
				finalModel.m_bottomOnAxis, finalModel.m_topOnAxis ) ;

			if( !isSucceed )
				printf(" cylinder edge 생성 실패. \n");

			*pOutLMOptCylinder = finalModel;
			pOutLMOptCylinder->SetAlgType( AM_LMOPT_TYPE );
			pOutLMOptCylinder->Normalize();
		}
	}

	//LM by ymcha
	finalModel = modelParam;
	AMLeastSquareFitting<CylinderPrimitive> LM;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	
	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}
	timeRefinement = stopWatch.LapSecond();
	
	if( finalModel.IsValidate() )
	{
		// cylinder의 boundary ( 축만 알면 됨 )
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, finalModel.m_axis, finalModel.m_center, 
			finalModel.m_bottomOnAxis, finalModel.m_topOnAxis ) ;

		if( !isSucceed )
			printf(" cylinder height 생성 실패. \n");

		//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
		pOutInlierIdxList->clear();
		pOutOutlierIdxList->clear();
		ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

		// cylinder가 너무 크다면 잘못구해진 것으로 간주하도록 한다.
		bool bCylinderDescribedWell = AMPtMath::IsDescribedCylinder( finalModel.m_axis, finalModel.m_center, finalModel.m_radius, pPointArray, pOutInlierIdxList );
		if( !bCylinderDescribedWell )
		{
			finalModel.m_radius = 0;
			pOutInlierIdxList->clear();
			pOutOutlierIdxList->clear();
			//printf(" data에 비해 cylinder 가 너무 커서 생성 실패. \n");
		}
	}
	else
	{
		pOutInlierIdxList->clear();
		pOutOutlierIdxList->clear();
	}

	timeAxisBoundary = stopWatch.LapSecond();
	stopWatch.Stop();
	timeTotal = stopWatch.ReadResultSecond();

	AMLog( "Cylinder fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Cylinder fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary %.3f)\n", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum, pOutInlierIdxList->size() );

//#define _AM_PRINT_FITTING_ERROR
#ifdef _AM_PRINT_FITTING_ERROR
	double distErrorAvg, distErrorStdDev;
	ransac.GetError( &finalModel, &distErrorAvg, &distErrorStdDev );
	printf( " dist error average = %f, stddev = %f\n", distErrorAvg, distErrorStdDev );
#endif
	return finalModel;
}

ConePrimitive AMFittingProject::FitCone( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold, 
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	ConePrimitive modelParam;
	int totalInputPointNum = 0;

	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMConeRANSAC<ConePrimitive> ransac;
	size_t  numCloseData = pPointArray->size();
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	//LM
	AMLeastSquareFitting<ConePrimitive> LM;

	ConePrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		//return modelParam;
		finalModel = modelParam;
	}

	// cone의 boundary ( 축만 알면 됨 )
	if( finalModel.IsValidate() )
	{
		bool isSucceed = AMPtMath::SetAxisBoundary( pPointArray, pOutInlierIdxList, finalModel.m_axis, finalModel.m_apex, 
			finalModel.m_bottomOnAxis, finalModel.m_topOnAxis ) ;

		if( !isSucceed )
			printf(" cone edge 생성 실패. \n");
	}
	
	if ( m_bVerbose ) 
	{
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	float timeAxisBoundary = stopWatch.LapSecond();
	timeTotal = stopWatch.StopSecond();
	
	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	pOutInlierIdxList->clear();
	pOutOutlierIdxList->clear();
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Cone fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary=  %.3f)", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	//printf( "Cone fitting: total %.3f sec (RANSAC %.3f, refining %.3f, axisBoundary=  %.3f)\n", timeTotal, timeRansac, timeRefinement, timeAxisBoundary );
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum, pOutInlierIdxList->size() );
	return finalModel;

}


TorusPrimitive AMFittingProject::FitTorus( vector<BMVector3f>* pPointArray, vector<BMVector3f>* pNormalArray, 
	float distThreshold, 	float angleThreshold,
	vector<int>* pOutInlierIdxList, vector<int>* pOutOutlierIdxList, vector<int>* pPointIdxList )
{
	TorusPrimitive modelParam;
	int totalInputPointNum = 0;

	if( pPointIdxList != nullptr )
	{
		totalInputPointNum = (int) pPointIdxList->size();
	}
	else
		totalInputPointNum = (int) pPointArray->size();

	if ( totalInputPointNum < MAX_REQUIRED_POINT_NUM )
	{
		//printf(" 피팅 작업을 수행하기에 포인트 수가 너무 적습니다.\n");
		return modelParam;
	}


	AMStopWatch stopWatch;
	float timeRansac = -1, timeRefinement = -1, timeTotal = -1;	// 걸린 시간 기록
	stopWatch.Start();
	Initialize();

	AMTorusRANSAC<TorusPrimitive> ransac;
	size_t numCloseData = pPointArray->size();
	ransac.SetDataPoints( pPointArray, pNormalArray, pPointIdxList );
	ransac.SetParameterWithAuto( 0.9f );
	ransac.SetErrorThreshold( distThreshold, angleThreshold );

	ransac.Evaluate( pOutInlierIdxList, pOutOutlierIdxList, &modelParam );
	timeRansac = stopWatch.LapSecond();

	if ( m_bVerbose ) {
		printf("[RANSAC Result]\n");
		modelParam.PrintParameter();
	}

	if( ! pOutInlierIdxList->size() ) 
	{
		//printf(" RANSAC 결과 inlier가 없습니다.\n");
		return modelParam;
	}

	//LM
	AMLeastSquareFitting<TorusPrimitive> LM;

	TorusPrimitive finalModel;
	int isSuccess = LM.MinimizeObjectiveFunction( &modelParam, pPointArray, pOutInlierIdxList, finalModel ); 
	timeRefinement = stopWatch.LapSecond();

	if( !isSuccess )
	{
		printf(" LM 실패\n");
		return modelParam;
	}

	if ( m_bVerbose ) {
		printf("[LM Result]\n");
		finalModel.PrintParameter();
	}

	timeTotal = stopWatch.StopSecond();

	//  normal error check 제외한 distance error 만을 고려하여 inlier, outlier 설정함(normal error 고려 안함)
	pOutInlierIdxList->clear();
	pOutOutlierIdxList->clear();
	ransac.GetInAndOutliers( &finalModel, pOutInlierIdxList, pOutOutlierIdxList, false );

	AMLog( "Torus fitting: total %.3f sec (RANSAC %.3f, refining %.3f)", timeTotal, timeRansac, timeRefinement );
	if ( AMGetFittingProject()->GetVerbose() )
		printf( "Torus fitting: total %.3f sec (RANSAC %.3f, refining %.3f)\n", timeTotal, timeRansac, timeRefinement );
	//printf(" Total Input Point Num %d Inlier Point Num %d \n", totalInputPointNum, pOutInlierIdxList->size() );

	return finalModel;
}

int	AMFittingProject::FitMultiPlane( std::vector<PlanePrimitive*>* pOutPlanePtrs, vector<BMVector3f>* pPointArray,
	vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitiveCount, bool isAxisAligned, float stopRatio )
{
	ASSERT( pOutPlanePtrs );
	ASSERT( pPointArray );
	ASSERT( maxPrimitiveCount > 0 );
	if ( !pOutPlanePtrs || !pPointArray ) return -1;
	if ( maxPrimitiveCount <= 0 ) return -1;
	pOutPlanePtrs->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;
	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );

	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pTobeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int) (pPointArray->size() * (1 - stopRatio));
	
	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		PlanePrimitive plane = this->FitPlane( pPointArray, pNormalArray, 
			distThreshold, angleThreshold, isAxisAligned, &inlierIndices, &outlierIndices, pTobeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Final Fitting Result ]\n" );
		plane.PrintParameter();

		FILE* pFile = fopen( "pca_test.txt", "w");
		for( int t = 0; t < inlierIndices.size(); t++)
		{
			int tIdx = inlierIndices.at(t);
			BMVector3f point = pPointArray->at( tIdx );
			fprintf( pFile, "%f %f %f \n", point.x, point.y, point.z );
		}
		fclose( pFile );	
#endif

		//AMGetPointProject()->m_bSavePrimitiveImg = false;	// 의미?
		if( plane.IsValidate() )
		{ 
			currentInlierIndices.swap( outlierIndices );
			pTobeUsedPointIndices = &currentInlierIndices;
			//AMGetPointProject()->m_primitiveManager.AddPrimitive( pNewPlane );
			PlanePrimitive* pNewPlane = new PlanePrimitive();
			*pNewPlane = plane;
			pOutPlanePtrs->push_back( pNewPlane );

			//AMGetPointProject()->SetInlierSecondColorOnPlane( pNewPlane, distThreshold, angleThreshold );
		}
		else
		{
			inlierIndices.clear();
			outlierIndices.clear();
			currentInlierIndices.clear();
			pTobeUsedPointIndices = nullptr;
			printf("Plane fitting 실패 (성공한 수: %d)\n", pOutPlanePtrs->size() ); 
			break;
		}

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
			//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutPlanePtrs->size() < maxPrimitiveCount && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutPlanePtrs->size();
}

int	AMFittingProject::FitMultiLine( std::vector<LinePrimitive*>* pOutLines, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutLines );
	ASSERT( pPointArray );
	ASSERT( maxPrimitives > 0 );
	if ( !pOutLines || !pPointArray ) return -1;
	if ( maxPrimitives <= 0 ) return -1;
	pOutLines->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int)(pPointArray->size() * (1 - stopRatio));

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		LinePrimitive line = this->FitLine( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Line Fitting Result ]\n" );
		line.PrintParameter();
#endif

		if( !line.IsValidate() )
		{ 
			printf("Line fitting 실패 (성공한 수: %d)\n", pOutLines->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		LinePrimitive* pNewLine = new LinePrimitive();
		*pNewLine = line;
		pOutLines->push_back( pNewLine );

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutLines->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutLines->size();
}

int	AMFittingProject::FitMultiLineOnXYPlane( std::vector<LinePrimitive*>* pOutLines, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutLines );
	ASSERT( pPointArray );
	if ( !pOutLines || !pPointArray ) return -1;
	
	vector<BMVector3f> projectedPointList;
	float	averageZ = 0;
	AMMathUtil::ProjectPointsOnXYPlane( &projectedPointList, &averageZ, *pPointArray );

	// XY plane상에서 fitting 수행
	FitMultiLine( pOutLines, &projectedPointList, pNormalArray, distThresholdParam, angleThresholdParam,
		maxPrimitives, stopRatio );

	// 원래대로 보내기
	BMMatrix44f transMat;
	AMMathUtil::MakeTranslationMatrix( BMVector3f(0.0, 0.0, averageZ), &transMat );

	// translate primtive to averageZ
	for ( int i = 0 ; i < pOutLines->size(); i++ )
	{
		IPrimitive* pPrimitive = pOutLines->at(i);
		LinePrimitive* pLinePrimitive = (LinePrimitive*)pPrimitive;
		pLinePrimitive->Translate( BMVector3f( 0, 0, averageZ ) );
	}

	return (int) pOutLines->size();
}

int	AMFittingProject::FitMultiCircle3D( std::vector<CirclePrimitive3D*>* pOutCircles, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutCircles );
	ASSERT( pPointArray );
	ASSERT( maxPrimitives > 0 );
	if ( !pOutCircles || !pPointArray ) return -1;
	if ( maxPrimitives <= 0 ) return -1;
	pOutCircles->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int) (pPointArray->size() * (1 - stopRatio));

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		CirclePrimitive3D circle = this->FitCircle3D( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Circle3D Fitting Result ]\n" );
		circle.PrintParameter();
#endif

		if( !circle.IsValidate() )
		{ 
			printf("Circle fitting 실패 (성공한 수: %d)\n", pOutCircles->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		CirclePrimitive3D* pNewCircle = new CirclePrimitive3D();
		*pNewCircle = circle;
		pOutCircles->push_back( pNewCircle );

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutCircles->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutCircles->size();
}

int	AMFittingProject::FitMultiSphere( std::vector<SpherePrimitive*>* pOutSpheres, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutSpheres );
	ASSERT( pPointArray );
	ASSERT( maxPrimitives > 0 );
	if ( !pOutSpheres || !pPointArray ) return -1;
	if ( maxPrimitives <= 0 ) return -1;
	pOutSpheres->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int)(pPointArray->size() * (1 - stopRatio));

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		SpherePrimitive sphere = this->FitSphere( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Sphere Fitting Result ]\n" );
		sphere.PrintParameter();
#endif

		if( !sphere.IsValidate() )
		{ 
			printf("Sphere fitting 실패 (성공한 수: %d)\n", pOutSpheres->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		SpherePrimitive* pNewSphere = new SpherePrimitive();
		*pNewSphere = sphere;
		pOutSpheres->push_back( pNewSphere );

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutSpheres->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutSpheres->size();
}

int	AMFittingProject::FitMultiCylinder( std::vector<CylinderPrimitive*>* pOutCylinders, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio,
	std::vector<CylinderPrimitive*>* pOutRANSACCylinders, std::vector<CylinderPrimitive*>* pOutLMOptCylinders )
{
	ASSERT( pOutCylinders );
	ASSERT( pPointArray );

	if ( !pOutCylinders || !pPointArray ) return -1;
	pOutCylinders->clear();
	if ( pOutRANSACCylinders ) pOutRANSACCylinders->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	/*if ( AMGetFittingProject() )
	{
	printf( "dist, angle threshold = %f, %f\n", distThreshold, angleThreshold );
	}*/

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int)(pPointArray->size() * (1 - stopRatio));
	if( maxPrimitives == 0 )
	{
		maxPrimitives = INT_MAX -1 ;
	}

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		if ( GetVerbose() )
		{
			printf( "FitCylinder: threshold(dist, angle) = (%f, %f)\n", distThreshold, angleThreshold );
		}

		CylinderPrimitive ransacCylinder;
		CylinderPrimitive lmOptCylinder; // lm opt version 
		CylinderPrimitive cylinder = this->FitCylinder( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices,
			&ransacCylinder, AMGetPointProject()->m_option.m_bTestLM_opt ? &lmOptCylinder : nullptr );
		

		if ( AMGetFittingProject()->GetVerbose() )
		{
			printf( "FitCylinder 결과, inlier = %d, outlier = %d\n", inlierIndices.size(), outlierIndices.size() );
		}

#ifdef _DEBUG
		printf("[ Cylinder Fitting Result ]\n" );
		cylinder.PrintParameter();
#endif

		if( !cylinder.IsValidate() )
		{ 
			printf("Cylinder fitting 실패 (성공한 수: %d)\n", pOutCylinders->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		CylinderPrimitive* pNewCylinder = new CylinderPrimitive();
		*pNewCylinder = cylinder;
		pOutCylinders->push_back( pNewCylinder );
		if ( pOutRANSACCylinders )
		{
			// to compare ransac and lm result
			CylinderPrimitive* pNewRANSACCylinder = new CylinderPrimitive();
			*pNewRANSACCylinder = ransacCylinder;
			pOutRANSACCylinders->push_back( pNewRANSACCylinder );
		}
		if( pOutLMOptCylinders )
		{
			// lm optimized version test
			CylinderPrimitive* pNewLMOptCylinder = new CylinderPrimitive();
			*pNewLMOptCylinder = lmOptCylinder;
			pOutLMOptCylinders->push_back( pNewLMOptCylinder );
		}
		if ( distThresholdParam == 0 )
		{
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
			if ( AMGetFittingProject()->GetVerbose() )
			{
				printf( "distThreshold -> %f\n", distThreshold );
			}
		}
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutCylinders->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum );

	return (int) pOutCylinders->size();
}

int	AMFittingProject::FitMultiCone( std::vector<ConePrimitive*>* pOutCones, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutCones );
	ASSERT( pPointArray );
	ASSERT( maxPrimitives > 0 );
	if ( !pOutCones || !pPointArray ) return -1;
	if ( maxPrimitives <= 0 ) return -1;
	pOutCones->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int)(pPointArray->size() * (1 - stopRatio));

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		ConePrimitive cone = this->FitCone( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Cone Fitting Result ]\n" );
		cone.PrintParameter();
#endif

		if( !cone.IsValidate() )
		{ 
			printf("Cone fitting 실패 (성공한 수: %d)\n", pOutCones->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		ConePrimitive* pNewCone = new ConePrimitive();
		*pNewCone = cone;
		pOutCones->push_back( pNewCone );

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutCones->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutCones->size();
}

int	AMFittingProject::FitMultiTorus( std::vector<TorusPrimitive*>* pOutTori, std::vector<BMVector3f>* pPointArray, 
	std::vector<BMVector3f>* pNormalArray, float distThresholdParam, float angleThresholdParam, int maxPrimitives, float stopRatio )
{
	ASSERT( pOutTori );
	ASSERT( pPointArray );
	ASSERT( maxPrimitives > 0 );
	if ( !pOutTori || !pPointArray ) return -1;
	if ( maxPrimitives <= 0 ) return -1;
	pOutTori->clear();

	float distThreshold = distThresholdParam;
	float angleThreshold = angleThresholdParam;

	// 자동 계산
	if ( distThresholdParam == 0 )
		distThreshold = GetAutoDistanceThreshold( pPointArray, 50 );
	if ( angleThreshold == 0 )
		angleThreshold = GetAutoAngleThreshold();

	vector<int> inlierIndices, outlierIndices, currentInlierIndices;
	vector<int>*	pToBeUsedPointIndices = nullptr;
	int	minimumInlierCount = 10;
	int stopAfterRemainingPointsNum = (int)(pPointArray->size() * (1 - stopRatio));

	do
	{
		inlierIndices.clear();
		outlierIndices.clear();

		TorusPrimitive torus = this->FitTorus( pPointArray, pNormalArray,
			distThreshold, angleThreshold, &inlierIndices, &outlierIndices, pToBeUsedPointIndices );

#ifdef _DEBUG
		printf("[ Torus Fitting Result ]\n" );
		torus.PrintParameter();
#endif

		if( !torus.IsValidate() )
		{ 
			printf("Torus fitting 실패 (성공한 수: %d)\n", pOutTori->size() ); 
			break;
		}

		currentInlierIndices.swap( outlierIndices );
		pToBeUsedPointIndices = &currentInlierIndices;
		TorusPrimitive* pNewTorus = new TorusPrimitive();
		*pNewTorus = torus;
		pOutTori->push_back( pNewTorus );

		//note: point index list에 pToBeUsedPointIndices를 넣어야 할 듯하지만 기존 구현 유지한다.
		if ( distThresholdParam == 0 )
			distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pToBeUsedPointIndices );
		//distThreshold = GetAutoDistanceThreshold( pPointArray, 50, pTobeUsedPointIndices );
	}
	while( pOutTori->size() < maxPrimitives && inlierIndices.size() > minimumInlierCount &&
		currentInlierIndices.size() > stopAfterRemainingPointsNum  );

	return (int) pOutTori->size();
}
