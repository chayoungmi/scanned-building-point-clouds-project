#include "stdafx.h"

#include "AMNormalEstimation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include "Common/AMMath.h"
#include "AMKNearestNeighbor.h"
#include "AMPoint2.h"
#include "AMOctree2.h"
#include "AMOctreeFile2.h"
#include "AMVisitorUtil.h"

#include "flann/flann.hpp"


using namespace std;

BMVector3f	AMEstimateNormalPCA( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor )
{
	ASSERT( kNearestNeighbor.m_kNearestPoints.size() >= 2 );
	if ( kNearestNeighbor.m_kNearestPoints.size() < 2 )
	{
		BMVector3f nanVector;
		nanVector.x = nanVector.y = nanVector.z = numeric_limits<float>::quiet_NaN();
		return nanVector;
	}
	
	cv::Mat	pcaInputPoints;
	pcaInputPoints.create( (int) kNearestNeighbor.m_kNearestPoints.size()+1, 3, CV_32F );

	AMKNearestNeighbor::KNearestPointList::const_iterator kNNIter = kNearestNeighbor.m_kNearestPoints.begin(),
		kNNIterEnd = kNearestNeighbor.m_kNearestPoints.end();

	// kNN에 자기 자신이 없으므로, 자신을 첫 포인트로 추가
	int row = 0;
	pcaInputPoints.at<float>( row, 0 ) = inputPoint.x;
	pcaInputPoints.at<float>( row, 1 ) = inputPoint.y;
	pcaInputPoints.at<float>( row, 2 ) = inputPoint.z;
	row++;
	for ( ; kNNIter != kNNIterEnd ; kNNIter++, row++ )
	{
		const BMVector3f& pos = kNNIter->second;
		pcaInputPoints.at<float>( row, 0 ) = pos.x;
		pcaInputPoints.at<float>( row, 1 ) = pos.y;
		pcaInputPoints.at<float>( row, 2 ) = pos.z;
	}

	cv::PCA pca( pcaInputPoints, cv::noArray(), CV_PCA_DATA_AS_ROW, 3 );
	// eigenvalue는 큰것->작은것 순서임

#ifdef _DEBUG
	static bool bFirstVisit = true;
	if ( bFirstVisit )
	{
		cout << "Input Points: " << pcaInputPoints << endl;
		cout << "Eigenvalues = " << pca.eigenvalues << endl;
		cout << "EigenVectors = " << pca.eigenvectors << endl;
		cout << "Mean = " << pca.mean << endl;
		bFirstVisit = false;
	}
#endif

	// 만약 component가 2개밖에 안 나온다면 실패로 간주
	// 3 point만 사용하는 경우에는 cross product 함수로 해야 함
	ASSERT( pca.eigenvalues.rows > 2 );
	if ( pca.eigenvalues.rows <= 2 )
	{
		AMLog( "%s(): PCA returned less than 2 components", __FUNCTION__ );

		BMVector3f nanVector;
		nanVector.x = nanVector.y = nanVector.z = numeric_limits<float>::quiet_NaN();
		return nanVector;
	}

	ASSERT( pca.eigenvalues.at<float>(0) >= pca.eigenvalues.at<float>(1) );
	ASSERT( pca.eigenvalues.at<float>(1) >= pca.eigenvalues.at<float>(2) );
#ifdef _DEBUG
//	cout << "eigen values = " << pca.eigenvalues << endl;
#endif

	BMVector3f normalVector;
	normalVector.x = pca.eigenvectors.at<float>(2,0);
	normalVector.y = pca.eigenvectors.at<float>(2,1);
	normalVector.z = pca.eigenvectors.at<float>(2,2);

	//BMVector3f vector1, normalVector;
	//vector1.x = pca.eigenvectors.at<float>(0,0);
	//vector1.y = pca.eigenvectors.at<float>(0,1);
	//vector1.z = pca.eigenvectors.at<float>(0,2);
	//BMVector3f vector2;
	//vector2.x = pca.eigenvectors.at<float>(1,0);
	//vector2.y = pca.eigenvectors.at<float>(1,1);
	//vector2.z = pca.eigenvectors.at<float>(1,2);

	//normalVector = vector1.CrossProduct(vector2);
	////normalVector.Normalize();

	return normalVector;
	//return vector2;
}

BMVector3f	AMEstimateNormalCrossProduct( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor )
{
	ASSERT( kNearestNeighbor.m_kNearestPoints.size() == 2 );

	ASSERT( kNearestNeighbor.m_kNearestPoints.size() >= 2 );
	if ( kNearestNeighbor.m_kNearestPoints.size() < 2 )
	{
		BMVector3f nanVector;
		nanVector.x = nanVector.y = nanVector.z = numeric_limits<float>::quiet_NaN();
		return nanVector;
	}

	AMKNearestNeighbor::KNearestPointList::const_iterator iter = kNearestNeighbor.m_kNearestPoints.begin();
	ASSERT( iter != kNearestNeighbor.m_kNearestPoints.end() );
	const BMVector3f& pos2 = (*iter).second;	iter++;
	ASSERT( iter != kNearestNeighbor.m_kNearestPoints.end() );
	const BMVector3f& pos3 = (*iter).second;

	BMVector3f v1 = pos2 - inputPoint;
	BMVector3f v2 = pos3 - inputPoint;

	BMVector3f normalVector = v1.CrossProduct( v2 );

	normalVector.Normalize();

	return normalVector;
}

#define NORMALESTIMATION_PRINTINFO_COUNTER 10000

void	AMEstimateNormalForPointArray( std::vector<BMVector3f>* pOutNormalArray, std::vector<BMVector3f>& inputPointArray, bool bVerbose )
{
	AMEstimateNormalWithFlann( pOutNormalArray, inputPointArray, bVerbose );
}

// 직접 구현한 KNN알고리즘을 사용하여 normal estimation 수행
void	AMEstimateNormalForPointArrayOld( std::vector<BMVector3f>* pOutNormalArray, const std::vector<BMVector3f>& inputPointArray, bool bVerbose )
{
	ASSERT( pOutNormalArray );
	if ( !pOutNormalArray ) return;
	pOutNormalArray->resize( inputPointArray.size() );

	vector<BMVector3f>::const_iterator inputIter = inputPointArray.cbegin(),
		inputIterEnd = inputPointArray.cend();

	int nearestNeighborToCount = 10;	// 10개의 NN을 사용
	int pointIndex = 0;

	if ( bVerbose )
	{
		printf( "Normal Estimation, points = %d\n", inputPointArray.size() );
	}

	int infoCounter = NORMALESTIMATION_PRINTINFO_COUNTER;
	for ( ; inputIter != inputIterEnd ; inputIter++, pointIndex++, infoCounter-- )
	{
		const BMVector3f& inputPoint = *inputIter;

		// kNN을 구한다.
		AMKNearestNeighbor kNN = AMCalculateKNNFromPointArray( inputPoint, inputPointArray, nearestNeighborToCount );

		// PCA를 이용하여 normal 계산
		BMVector3f normalVector = AMEstimateNormalPCA( inputPoint, kNN );

		pOutNormalArray->at( pointIndex ) = normalVector;

		if ( bVerbose && infoCounter <= 0 )
		{
			infoCounter = NORMALESTIMATION_PRINTINFO_COUNTER;
			printf( "Number of points processed = %d (%f %%)\n", pointIndex,
				pointIndex / (float) inputPointArray.size() * 100 );
		}
	}
}

void	AMEstimateNormalForPointArrayOld( std::vector<BMVector3f>* pOutNormalArray,
	const std::vector<AMPoint2>& inputPointArrayAMPoint2, bool bVerbose )
{
	// AMPoint2 array를 BMVector3 array로 변경
	vector<BMVector3f> inputPointArray;
	inputPointArray.resize( inputPointArrayAMPoint2.size() );
	for ( int i = 0 ; i < inputPointArrayAMPoint2.size(); i++ )
	{
		inputPointArray[i] = inputPointArrayAMPoint2[i].m_pos;
	}

	ASSERT( pOutNormalArray );
	if ( !pOutNormalArray ) return;
	pOutNormalArray->resize( inputPointArray.size() );

	vector<BMVector3f>::const_iterator inputIter = inputPointArray.cbegin(),
		inputIterEnd = inputPointArray.cend();

	int nearestNeighborToCount = 10;	// 10개의 NN을 사용
	int pointIndex = 0;

	if ( bVerbose ) printf( "Normal estimation: points = %d\n", inputPointArrayAMPoint2.size() );

	int infoCounter = NORMALESTIMATION_PRINTINFO_COUNTER;
	for ( ; inputIter != inputIterEnd ; inputIter++, pointIndex++, infoCounter-- )
	{
		const BMVector3f& inputPoint = *inputIter;

		// kNN을 구한다.
		AMKNearestNeighbor kNN = AMCalculateKNNFromPointArray( inputPoint, inputPointArray, nearestNeighborToCount );

		// PCA를 이용하여 normal 계산
		BMVector3f normalVector = AMEstimateNormalPCA( inputPoint, kNN );

		pOutNormalArray->at( pointIndex ) = normalVector;

		if ( bVerbose && infoCounter <= 0 )
		{
			infoCounter = NORMALESTIMATION_PRINTINFO_COUNTER;
			printf( "Number of points processed = %d (%f %%)\n", pointIndex,
				pointIndex / (float) inputPointArray.size()* 100 );
		}
	}
}

//#define _AM_DEBUG_NORMAL_ESTIMATION

bool	AMEstimateNormalWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<AMPoint2>& inputPointArray, bool bVerbose )
{

#ifdef _AM_DEBUG_NORMAL_ESTIMATION
	cout << "Num input = " << inputPointArray.size() << endl;
	for ( int abc = 0; abc < inputPointArray.size(); abc++ )
	{
		AMPoint2 pt = inputPointArray.at(abc);

		cout <<  pt.m_pos.x << ", " << pt.m_pos.y << ", " << pt.m_pos.z << endl;
	}
	cout << endl;
#endif // _AM_DEBUG_NORMAL_ESTIMATION

	ASSERT( pOutNormalArray );
	if ( !pOutNormalArray ) return false;
	int numPoints = (int) inputPointArray.size();

	AMStopWatch stopWatch;
	stopWatch.Start();

	vector<int>	indexArray;
	vector<float>	distArray;
	int nearestNeighborToCount = 10;	// 10개의 NN 사용
	try {
		pOutNormalArray->resize( inputPointArray.size() );
		if ( numPoints == 0 ) return true;
		
		indexArray.resize( numPoints * nearestNeighborToCount );
		distArray.resize( numPoints * nearestNeighborToCount );
	}
	catch ( ... )
	{
		cerr << "Normal estimation: memory not enough, point count = " << numPoints << endl;
		AMLog( "Normal estimation: memory not enough, point count = ", numPoints);
		return false;
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "memory allocation time: " << lapTime << endl;
		AMLog( "Normal estimation: memory allocation time = %f sec\n", lapTime );
	}

	flann::Matrix<int>	indexMatrix( &indexArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	distMatrix( &distArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	inputMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(AMPoint2) );
	flann::Matrix<float>	queryMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(AMPoint2) );

	flann::Index< flann::L2<float> > kdTree( inputMatrix, flann::KDTreeIndexParams(1) );
	kdTree.buildIndex();

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "Index building time: " << lapTime << endl;
		AMLog( " index building time: %f", lapTime );
	}

	kdTree.knnSearch( queryMatrix, indexMatrix, distMatrix, nearestNeighborToCount, flann::SearchParams(-1) );

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " knnSearch time: " << lapTime << endl;
		AMLog( " knnSearch time: %f", lapTime );
	}

	// NN 구조 빌드 완료

	AMKNearestNeighbor kNN( nearestNeighborToCount );	// 데이터 맞추기 위한 것
	for ( int i = 0; i < numPoints; i++ )
	{
		kNN.m_kNearestPoints.clear();
		// note: k=0이면 자기 자신이 돌아와서 뺐음
		for ( int k = 1 ; k < nearestNeighborToCount; k++ )
		{
			int index = i * nearestNeighborToCount + k;
			float dist = distArray.at( index );
			int outPointIndex = indexArray.at( index );
			
			kNN.m_kNearestPoints.insert( AMKNearestNeighbor::KNearestPointList::value_type(
				dist, inputPointArray.at( outPointIndex ).m_pos ) );

#ifdef _AM_DEBUG_NORMAL_ESTIMATION
			cout << outPointIndex << " ";
#endif // _AM_DEBUG_NORMAL_ESTIMATION
		}
		BMVector3f normalVector = AMEstimateNormalPCA( inputPointArray.at(i).m_pos, kNN );
		pOutNormalArray->at(i) = normalVector;
#ifdef _AM_DEBUG_NORMAL_ESTIMATION
		cout << " normal = " << normalVector << endl;
		cout << endl;
#endif
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " PCA normal estimation time: " << lapTime << "secs" << endl;
		AMLog( "PCA normal estimation time: %f secs", lapTime );
	}

	int durationMillis = stopWatch.Stop();
	if ( bVerbose )
	{
		float totalTime = durationMillis / 1000.0f;
		cout << "Normal estimation, total time = " << totalTime << endl;
	}

	return true;
}

// 
bool	AMEstimateNormalWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<BMVector3f>& inputPointArray, bool bVerbose )
{
	ASSERT( pOutNormalArray );
	if ( !pOutNormalArray ) return false;
	int numPoints = (int) inputPointArray.size();

	AMStopWatch stopWatch;
	stopWatch.Start();

	vector<int>	indexArray;
	vector<float>	distArray;
	int nearestNeighborToCount = 10;	// 10개의 NN 사용
	try {
		pOutNormalArray->resize( inputPointArray.size() );
		if ( numPoints == 0 ) return true;

		indexArray.resize( numPoints * nearestNeighborToCount );
		distArray.resize( numPoints * nearestNeighborToCount );
	}
	catch ( ... )
	{
		cerr << "Normal estimation: memory not enough, point count = " << numPoints << endl;
		AMLog( "Normal estimation: memory not enough, point count = ", numPoints);
		return false;
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "memory allocation time: " << lapTime << endl;
		AMLog( "Normal estimation: memory allocation time = %f sec\n", lapTime );
	}

	flann::Matrix<int>	indexMatrix( &indexArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	distMatrix( &distArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	inputMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );
	flann::Matrix<float>	queryMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );

	flann::Index< flann::L2<float> > kdTree( inputMatrix, flann::KDTreeIndexParams(1) );
	kdTree.buildIndex();

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "Index building time: " << lapTime << endl;
		AMLog( " index building time: %f", lapTime );
	}

	kdTree.knnSearch( queryMatrix, indexMatrix, distMatrix, nearestNeighborToCount, flann::SearchParams(-1) );

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " knnSearch time: " << lapTime << endl;
		AMLog( " knnSearch time: %f", lapTime );
	}

	// NN 구조 빌드 완료

	AMKNearestNeighbor kNN( nearestNeighborToCount );	// 데이터 맞추기 위한 것
	for ( int i = 0; i < numPoints; i++ )
	{
		kNN.m_kNearestPoints.clear();
		// note: k=0이면 자기 자신이 돌아와서 뺐음
		for ( int k = 1 ; k < nearestNeighborToCount; k++ )
		{
			int index = i * nearestNeighborToCount + k;
			float dist = distArray.at( index );
			int outPointIndex = indexArray.at( index );

			kNN.m_kNearestPoints.insert( AMKNearestNeighbor::KNearestPointList::value_type(
				dist, inputPointArray.at( outPointIndex ) ) );
		}

		BMVector3f normalVector = AMEstimateNormalPCA( inputPointArray.at(i), kNN );
		pOutNormalArray->at(i) = normalVector;
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " PCA normal estimation time: " << lapTime << "secs" << endl;
		AMLog( "PCA normal estimation time: %f secs", lapTime );
	}

	int durationMillis = stopWatch.Stop();
	if ( bVerbose )
	{
		float totalTime = durationMillis / 1000.0f;
		cout << "Normal estimation, total time = " << totalTime << endl;
	}

	return true;
}

bool	AMEstimateNormalForPointArrayWithOctree( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<AMPoint2>* pInOutPointArray, bool bVerbose )
{
	ASSERT( pOutNormalArray && pInOutPointArray );
	if ( !pOutNormalArray || !pInOutPointArray ) return false;

	DWORD t1 = ::timeGetTime();

	// 1. BB 계산
	AMOctreeBBox3f bbox;
	vector<AMPoint2>::iterator iter = pInOutPointArray->begin(), iterEnd = pInOutPointArray->end();
	for ( ; iter != iterEnd; iter++ )
	{
		AMPoint2& pt = *iter;
		bbox.UpdateBoundingBox( pt.m_pos );
	}

	DWORD t2 = ::timeGetTime();
	printf( "BB계산, %.3f sec\n", (t2-t1) / 1000.0f );

	// 2. Octree 생성
	AMOctree2* pTempOctree = new AMOctree2();
	int splitSize = 30001;
//	tempOctree.CreateInMemoryOctree( bbox, splitSize );
	if ( !pTempOctree->Create( "TempOctree.ampc", "TempOctree.ampc", bbox, splitSize, false ) )
	{
		ASSERT( !"tempOctree.Create() failed" );
		return false;
	}

	for ( iter = pInOutPointArray->begin();
		iter != iterEnd;
		iter++ )
	{
		AMPoint2& pt = *iter;

		pTempOctree->Add( pt );
	}

	bool bSuccess = pTempOctree->SaveOctreeNodes();
	ASSERT( bSuccess );
	bSuccess = pTempOctree->SavePointData() && bSuccess;
	ASSERT( bSuccess );
	pTempOctree->m_pPointStorage->Flush();
	delete pTempOctree;

	pTempOctree = new AMOctree2();
	pTempOctree->Load( "TempOctree.ampc" );

	DWORD t3 = ::timeGetTime();
	printf( "Octree생성, %.3f sec\n", (t3-t2) / 1000.0f );

	AMForceLoadPointVisitor forceLoadVisitor(1.0f);
	forceLoadVisitor.AddProperty( "normal" );
	pTempOctree->Visit( &forceLoadVisitor, nullptr, OCTREEVISIT_NODE, 1, nullptr );

	DWORD t4 = ::timeGetTime();
	printf( "force load, %.3f sec\n", (t4-t3) / 1000.0f );

	// NOTE: GenerateNormalData에서 생성도 처리한다.
	//if ( !tempOctree.CreatePropertyFile( "normal", "normal", BMVector3f(0, 0, 0), true ) )
	//{
	//	ASSERT( !"cannot create normal file" );
	//	tempOctree.Clear();
	//	return false;
	//}

	//pTempOctree->GenerateNormalData( 1.0f, 1.0f );
	pTempOctree->_GenerateNormalDataInCore();

	DWORD t5 = ::timeGetTime();
	printf( "Normal estimation, %.3f sec\n", (t5-t4) / 1000.0f );

	int pointCount = (int) pInOutPointArray->size();
	pInOutPointArray->clear();
	pInOutPointArray->reserve( pointCount );
	pOutNormalArray->reserve( pointCount );

	// 생성된 데이터와 point list를 가져온다.
	AMLoadNormalAndPointPropertyVisitor	loadNormalAndPointVisitor( pOutNormalArray, pInOutPointArray );
	pTempOctree->Visit( &loadNormalAndPointVisitor, nullptr, OCTREEVISIT_INCORE_ALL, 1.0f, nullptr );

	ASSERT( pInOutPointArray->size() == loadNormalAndPointVisitor.m_pNormalList->size() );
	ASSERT( pInOutPointArray->size() == pointCount );

	delete pTempOctree;
	return true;
}

//----------------------------------------------------------------------------------
// AMCollectSampleNeighborsWithFlann
// : sample 중심으로 모든 points 에 대해 knn 구해주는 것임
// output	- pOutNormalArray : normal value array
// output	- pOutSampleNeighborIdxArray : neighbor indices, neighbor indices, ...
//	input	- inputPointArray : input point array
//	input	- samplePointIdxArray : sample point indices in input point array
//	input	- bGetNormal		: if want to get normal, set value to true

bool	AMCollectSampleNeighborsWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<int>* pOutSampleNeighborIdxArray, std::vector<BMVector3f>& inputPointArray,
	std::vector<int>& samplePointIdxArray, int nearestNeighborToCount, bool bGetNormal, bool bVerbose )
{
	ASSERT( pOutNormalArray );
	ASSERT( pOutSampleNeighborIdxArray );
	if ( !pOutNormalArray ) return false;
	if ( !pOutSampleNeighborIdxArray ) return false;
	int numPoints = (int) inputPointArray.size();

	AMStopWatch stopWatch;
	stopWatch.Start();

	vector<int>	indexArray;
	vector<float>	distArray;

	try {
		pOutNormalArray->resize( inputPointArray.size() );
		pOutSampleNeighborIdxArray->resize( samplePointIdxArray.size() * (nearestNeighborToCount) );
		if ( numPoints == 0 ) return true;

		indexArray.resize( numPoints * nearestNeighborToCount );
		distArray.resize( numPoints * nearestNeighborToCount );
	}
	catch ( ... )
	{
		cerr << "Normal estimation: memory not enough, point count = " << numPoints << endl;
		AMLog( "Normal estimation: memory not enough, point count = ", numPoints);
		return false;
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "memory allocation time: " << lapTime << endl;
		AMLog( "Normal estimation: memory allocation time = %f sec\n", lapTime );
	}

	flann::Matrix<int>		indexMatrix( &indexArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	distMatrix( &distArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	inputMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );
	flann::Matrix<float>	queryMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );

	flann::Index< flann::L2<float> > kdTree( inputMatrix, flann::KDTreeIndexParams(1) );
	kdTree.buildIndex();

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "Index building time: " << lapTime << endl;
		AMLog( " index building time: %f", lapTime );
	}

	kdTree.knnSearch( queryMatrix, indexMatrix, distMatrix, nearestNeighborToCount, flann::SearchParams(-1) );

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " knnSearch time: " << lapTime << endl;
		AMLog( " knnSearch time: %f", lapTime );
	}

	// 모든 포인트들에 대한 normal 구한다.
	if( bGetNormal)
	{
		// NN 구조 빌드 완료
		AMKNearestNeighbor kNN( nearestNeighborToCount );	// 데이터 맞추기 위한 것

		for ( int i = 0; i < numPoints; i++ )
		{
			kNN.m_kNearestPoints.clear();
			// note: k=0이면 자기 자신이 돌아와서 뺐음
			for ( int k = 1 ; k < nearestNeighborToCount; k++)
			{
				int index = i * nearestNeighborToCount + k;
				float dist = distArray.at( index );
				int outPointIndex = indexArray.at( index );

				kNN.m_kNearestPoints.insert( AMKNearestNeighbor::KNearestPointList::value_type(
					dist, inputPointArray.at( outPointIndex ) ) );
			}

			BMVector3f normalVector = AMEstimateNormalPCA( inputPointArray.at(i), kNN );
			pOutNormalArray->at(i) = normalVector;
		}
	}
	
	// samples 에 대해 저장한다.
	
	int neighbirCnt = 0;
	for( int i = 0; i < samplePointIdxArray.size(); i++ )
	{
		int sampleIdx = samplePointIdxArray.at(i);

		for( int k = 0; k < nearestNeighborToCount; k++ )
		{
			int neighborIdx = sampleIdx * nearestNeighborToCount + k ;
			int index = indexArray.at( neighborIdx );
			pOutSampleNeighborIdxArray->at( neighbirCnt ) = index ;
			neighbirCnt++;
		}
	}
	
	if ( bVerbose && bGetNormal )
	{
		float lapTime = stopWatch.LapSecond();
		cout << " PCA normal estimation time: " << lapTime << "secs" << endl;
		AMLog( "PCA normal estimation time: %f secs", lapTime );
	}

	int durationMillis = stopWatch.Stop();
	if ( bVerbose && bGetNormal )
	{
		float totalTime = durationMillis / 1000.0f;
		cout << "Normal estimation, total time = " << totalTime << endl;
	}

	return true;
}

// 
bool	AMMakeKnnWithFlann( std::vector<BMVector3f>& inputPointArray, 
					std::vector<int>* pOutArray, bool bVerbose )
{
	ASSERT( pOutArray ) ;
	if ( !pOutArray ) return false;
	int numPoints = (int) inputPointArray.size();

	AMStopWatch stopWatch;
	stopWatch.Start();

	vector<int>	indexArray;
	vector<float>	distArray;
	int nearestNeighborToCount = 10;	// 10개의 NN 사용
	try {
		pOutArray->resize( inputPointArray.size() * nearestNeighborToCount );
		if ( numPoints == 0 ) return true;

		indexArray.resize( numPoints * nearestNeighborToCount );
		distArray.resize( numPoints * nearestNeighborToCount );
	}
	catch ( ... )
	{
		cerr << "Make Knn: memory not enough, point count = " << numPoints << endl;
		AMLog( "Make Knn: memory not enough, point count = ", numPoints);
		return false;
	}

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "memory allocation time: " << lapTime << endl;
		AMLog( "Normal estimation: memory allocation time = %f sec\n", lapTime );
	}

	flann::Matrix<int>		indexMatrix( &indexArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	distMatrix( &distArray[0], numPoints, nearestNeighborToCount );
	flann::Matrix<float>	inputMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );
	flann::Matrix<float>	queryMatrix( (float*) &inputPointArray[0], numPoints, 3, sizeof(BMVector3f) );

	flann::Index< flann::L2<float> > kdTree( inputMatrix, flann::KDTreeIndexParams(1) );
	kdTree.buildIndex();

	if ( bVerbose )
	{
		float lapTime = stopWatch.LapSecond();
		cout << "Index building time: " << lapTime << endl;
		AMLog( " index building time: %f", lapTime );
	}

	kdTree.knnSearch( queryMatrix, indexMatrix, distMatrix, nearestNeighborToCount, flann::SearchParams(-1) );

	if ( bVerbose )
	{ 
		float lapTime = stopWatch.LapSecond();
		cout << " knnSearch time: " << lapTime << endl;
		AMLog( " knnSearch time: %f", lapTime );
	}

	int outIndex = 0;
	for ( int i = 0; i < numPoints; i++ )
	{
		for ( int k = 0 ; k < nearestNeighborToCount; k++ )
		{
			int index = i * nearestNeighborToCount + k;
			int outPointIndex = indexArray.at( index );

			pOutArray->at(outIndex) = outPointIndex;
			outIndex++;
		}
	}

	return true;
}