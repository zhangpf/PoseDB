/*
Copyright (c) 2016, Pengfei Zhang<zpfalpc23@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <rtabmap/utilite/ULogger.h>

#include "rtabmap/core/VWDictionary.h"
#include "rtabmap/core/VisualWord.h"

#include "rtabmap/core/Signature.h"
#include "rtabmap/core/DBDriver.h"
#include "rtabmap/core/Parameters.h"

#include "rtabmap/utilite/UtiLite.h"
#include "rtflann/flann.hpp"

#include <opencv2/opencv_modules.hpp>

#include "rtabmap/core/PoseSpace.h"

#include <fstream>
#include <string>

namespace rtabmap {

PoseSpace::PoseSpace() {
	index_ = NULL;
	_mapIndexId.clear();
	_mapIdIndex.clear();
	nextIndex_ = 0;
	_removed.clear();
}

bool PoseSpace::addPoint(const Transform & pose, const int signId) {
	float p[6];	
	int idx = nextIndex_;

	rtflann::Index<rtflann::L2<float> > * index = (rtflann::Index<rtflann::L2<float> > *)index_;


	if (_removed.find(signId) == _removed.end() && _mapIdIndex.find(signId) != _mapIdIndex.end()) {
		return false;
	}
	if (_removed.find(signId) != _removed.end()) {
		_removed.erase(signId);
	}
	nextIndex_ ++;
	std::pair<std::map<int, int>::iterator, bool> inserted;
	pose.getTranslationAndEulerAngles(p[0], p[1], p[2], p[3], p[4], p[5]);
	rtflann::Matrix<float> point(p, 1, 6);

	if (!index_) {
		rtflann::IndexParams params = rtflann::KDTreeIndexParams();
		index = new rtflann::Index<rtflann::L2<float> >(point, params);
		index->buildIndex();
		index_ = index;
	} else {
		index->addPoints(point);
	}

	inserted = _mapIndexId.insert(std::pair<int, int>(idx, signId));
	//UASSERT(inserted.second);
	inserted = _mapIdIndex.insert(std::pair<int, int>(signId, idx));
	//UASSERT(inserted.second);
	return true;
}

void PoseSpace::removePoint(const int signId) {
	_removed.insert(signId);
}

std::list<int> PoseSpace::findNN(const Transform & pose, int k) {
	float p[6];
	std::list<int> resultIds;
	rtflann::Index<rtflann::L2<float> > * index = (rtflann::Index<rtflann::L2<float> > *)index_;
	
	pose.getTranslationAndEulerAngles(p[0], p[1], p[2], p[3], p[4], p[5]);
	rtflann::Matrix<float> query(p, 1, 6);
	std::vector< std::vector<int> > indice(k);
    std::vector<std::vector<float> > dists(k);

	index->knnSearch(query, indice, dists, k, rtflann::SearchParams());

	for (size_t i = 0; i < indice[0].size(); i ++) {
		int id = uValue(_mapIndexId, indice[0][i]);
		if (_removed.find(id) != _removed.end()) continue;
		resultIds.push_back(id);
	}
	return resultIds;
}

}