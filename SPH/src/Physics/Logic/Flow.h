#pragma once
#include <stdlib.h>
#include <time.h>  

#include "Core/Base.h"
#include "Core/CommonTypes.h"

#include "Physics/Entities/Particles.h"

class Flow
{

public:
	Flow(flow_area in_area, flow_area out_area):in_area(in_area), out_area(out_area) 
	{
		srand(time(NULL));
	}
	
	void in(size_t amount, Ref<Particles> particles) const;
	
	void out(Ref<Particles> particles) const;


	~Flow() {}

private:
	float random(float from, float to) const;

private:
	flow_area in_area;
	flow_area out_area;
};

