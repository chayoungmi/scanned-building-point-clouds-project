#pragma once

#include "AMKNearestNeighbor.h"
#include "AMPoint2.h"

class AMKNearestNeighbor;
class AMUndirectedKNNGraph;

BMVector3f	AMEstimateNormalPCA( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor );

BMVector3f	AMEstimateNormalCrossProduct( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor );

/** Point�� point�� ���� K nearest neighbor �����͸� �̿��Ͽ� normal�� estimate�Ѵ�. */
inline
BMVector3f	AMEstimateNormal( const BMVector3f& inputPoint, const AMKNearestNeighbor& kNearestNeighbor )
{
	if ( kNearestNeighbor.m_k == 2 )
		return AMEstimateNormalCrossProduct( inputPoint, kNearestNeighbor );
	else
		return AMEstimateNormalPCA( inputPoint, kNearestNeighbor );
}

//////////////////////////////////////////////////////////////////////////
// Point array�� ���� normal estimation.
// Octree�� ���� AMOctree���� ó��
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

// Normal estimation�� �����Ѵ�. point array�� octree�� ���� ���ġ�ȴ�.
bool	AMEstimateNormalForPointArrayWithOctree( std::vector<BMVector3f>* pOutNormalArray,
	std::vector<AMPoint2>* pInOutPointArray, bool bVerbose = false );

bool	AMMakeKnnWithFlann( std::vector<BMVector3f>& inputPointArray, 
	std::vector<int>* pOutArray, bool bVerbose );