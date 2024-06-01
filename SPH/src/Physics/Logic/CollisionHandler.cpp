#include "CollisionHandler.h"


void CollisionHandler::Resolve(Ref<Particles> particles, float deltaTime) const {
	float halfWidth = p_spec.Width / 2.0f;
	float halfHeight = p_spec.Height / 2.0f;

	float min_x = -halfWidth + p_spec.ParticleRadius;
	float max_x = halfWidth - p_spec.ParticleRadius;
	float min_y = -halfHeight + p_spec.ParticleRadius;
	float max_y = halfHeight - p_spec.ParticleRadius;

	UpdateAndCollisionCuda(particles->getSize(), deltaTime, -(1.0f-p_spec.CollisionDamping), p_spec.ParticleRadius,
		min_x, max_x, min_y, max_y, c_obstacles_addr, obstacles_size);

}