#pragma once

#include "AMKNearestNeighbor.h"
#include "AMPoint2.h"

class AMKNearestNeighbor;
class AMUndirectedKNNGraph;

BMVector3f	AMEstimateNormalPCA( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor );

BMVector3f	AMEstimateNormalCrossProduct( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor );

/** Point의 point에 대해 K nearest neighbor 데이터를 이용하여 normal을 estimate한다. */
inline
BMVector3f	AMEstimateNormal( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor )
{
	if ( kNearestNeighbor.m_k == 2 )
		return AMEstimateNormalCrossProduct( inputPoint, kNearestNeighbor );
	else
		return AMEstimateNormalPCA( inputPoint, kNearestNeighbor );
}

//////////////////////////////////////////////////////////////////////////
// Point array에 대한 normal estimation.
// Octree의 경우는 AMOctree에서 처리
void	AMEstimateNormalForPointArray( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<BMVector3f>& inputPointArray, bool bVerbose = false );
//void	AMEstimateNormalForPointArray( std::vector<BMVector3f>* pOutNormalArray,
//	const std::vector<AMPoint2>& inputPointArray, bool bVerbose = false );

bool	AMEstimateNormalWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<BMVector3f>& inputPointArray, bool bVerbose = false );
bool	AMEstimateNormalWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<AMPoint2>& inputPointArray, bool bVerbose = false );

// normal estimation and construct sample point knn 
bool	AMCollectSampleNeighborsWithFlann( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<int>* pOutSampleNeighborIdxArray, std::vector<BMVector3f>& inputPointArray,
	std::vector<int>& samplePointIdxArray, int nearestNeighborToCount, bool bGetNormal, bool bVerbose = false );

// Normal estimation을 수행한다. point array는 octree에 의해 재배치된다.
bool	AMEstimateNormalForPointArrayWithOctree( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<AMPoint2>* pInOutPointArray, bool bVerbose = false );

bool	AMMakeKnnWithFlann( std::vector<BMVector3f>& inputPointArray, 
	std::vector<int>* pOutArray, bool bVerbose );