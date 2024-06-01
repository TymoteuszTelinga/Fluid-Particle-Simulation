#pragma once
#include <glm/glm.hpp>

/**
* Class for managing every particle
*
* @note uses CUDA 
*/
class Particles
{
public:
	/**
	* The constructor.
	* 
	* @note uses CUDA
	* 
	* @param capacity maximum capacity of prticles defined by user, must be less or equal to maxSize
	* @param maxSize hard limit of amount of particles in simulation, it nust not be exceeded 
	*/
	Particles(size_t capacity, size_t maxSize);

	/**
	* The desctructor
	* 
	* @note uses CUDA
	*/
	~Particles();

	/**
	* adds new particle with passed position only if amount is lower than capacity
	* 
	* @param x_pos horizontal position of particle
	* @param y_pos vertical position of particle
	* @return true if particle was added, otherwise false
	*/
	bool addParticle(float x_pos, float y_pos);

	/**
	* returns current amount of generated particles
	* 
	* @return amount of particles
	*/

	size_t getSize();

	/**
	* return maximum capacity of particles  
	* 
	* @return capacity of particles
	*/
	size_t getCapacity();

	/**
	* updates maximum capacity, it will not exceeds hard limit of amount of particles
	* 
	* @param newCapacity new capacity
	*/
	void setCapacity(size_t newCapacity);

	/**
	* returns position of selected particle
	* 
	* @param index of particle
	* @return position of particle or position (0,0) if index exceeds size
	*/
	glm::vec2 getPosition(size_t index);

	/**
	* returns predicted position of selected particle
	* 
	* @param index of particle
	* @return predicted position of particle or position (0,0) if index exceeds size
	*/
	glm::vec2 getPredictedPosition(size_t index);

	/**
	* calculates and returns distance between predicted postions of two particles
	* 
	* @param firstIndex index of first particle
	* @param secondIndex index of second particle
	* @return distance between particles or zero if any of indexes exceeds size
	*/
	float calculatePredictedDistance(size_t firstIndex, size_t secondIndex);

	/**
	* removes selected particle
	* 
	* Removal works by swapping sleected particle with particle at the end (at size-1 position)
	* then size is decreased by one
	* 
	* @param index of selected particle
	*/
	void remove(size_t index);

	/**
	* trnasfers particles' positions and velocities to GPU memory.
	*/
	void sendToCuda();

	/**
	* transfers particles' positons and velocities from GPU memory.
	*/
	void getFromCuda();

	/**
	* transfers particles' predicted positions from GPU memory.
	*/
	void getFromCudaBeforeSpatial();
		

private:
	const size_t maxSize;
	size_t capacity;
	size_t size;

	float* positions_x;
	float* positions_y;

	float* velocities_x;
	float* velocities_y;

	float* predictedPositions_x;
	float* predictedPositions_y;

public:
	float* c_positions_x_addr;
	float* c_positions_y_addr;
	float* c_velocities_x_addr;
	float* c_velocities_y_addr;
	float* c_pred_positions_x_addr;
	float* c_pred_positions_y_addr;
	int* c_lookup_indexes_addr;
	int* c_lookup_keys_addr;
	int* c_indices_addr;
	int c_indices_size;

};

