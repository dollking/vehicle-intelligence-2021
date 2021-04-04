import numpy as np
from helpers import distance

class ParticleFilter:
    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles

    # Set the number of particles.
    # Initialize all the particles to the initial position
    #   (based on esimates of x, y, theta and their uncertainties from GPS)
    #   and all weights to 1.0.
    # Add Gaussian noise to each particle.
    def initialize(self, x, y, theta, std_x, std_y, std_theta):
        self.particles = []
        for i in range(self.num_particles):
            self.particles.append({
                'x': np.random.normal(x, std_x),
                'y': np.random.normal(y, std_y),
                't': np.random.normal(theta, std_theta),
                'w': 1.0,
                'assoc': [],
            })
        self.initialized = True

    # Add measurements to each particle and add random Gaussian noise.
    def predict(self, dt, velocity, yawrate, std_x, std_y, std_theta):
        # Be careful not to divide by zero.
        v_yr = velocity / yawrate if yawrate else 0
        yr_dt = yawrate * dt
        for p in self.particles:
            # We have to take care of very small yaw rates;
            #   apply formula for constant yaw.
            if np.fabs(yawrate) < 0.0001:
                xf = p['x'] + velocity * dt * np.cos(p['t'])
                yf = p['y'] + velocity * dt * np.sin(p['t'])
                tf = p['t']
            # Nonzero yaw rate - apply integrated formula.
            else:
                xf = p['x'] + v_yr * (np.sin(p['t'] + yr_dt) - np.sin(p['t']))
                yf = p['y'] + v_yr * (np.cos(p['t']) - np.cos(p['t'] + yr_dt))
                tf = p['t'] + yr_dt

            p['x'] = np.random.normal(xf, std_x)
            p['y'] = np.random.normal(yf, std_y)
            p['t'] = np.random.normal(tf, std_theta)


    # Find the predicted measurement that is closest to each observed
    #   measurement and assign the observed measurement to this
    #   particular landmark.
    def associate(self, predicted, observations):
        associations = []
        # For each observation, find the nearest landmark and associate it.
        #   You might want to devise and implement a more efficient algorithm.
        for o in observations:
            min_dist = -1.0
            for p in predicted:
                dist = distance(o, p)
                if min_dist < 0.0 or dist < min_dist:
                    min_dist = dist
                    min_id = p['id']
                    min_x = p['x']
                    min_y = p['y']
            association = {
                'id': min_id,
                'x': min_x,
                'y': min_y,
            }
            associations.append(association)
        # Return a list of associated landmarks that corresponds to
        #   the list of (coordinates transformed) predictions.
        return associations

    # Update the weights of each particle using a multi-variate
    #   Gaussian distribution.
    def update_weights(self, sensor_range, std_landmark_x, std_landmark_y,
                       observations, map_landmarks):
        import math
        # TODO: For each particle, do the following:
        # 1. Select the set of landmarks that are visible
        #    (within the sensor range).
        # 2. Transform each observed landmark's coordinates from the
        #    particle's coordinate system to the map's coordinates.
        # 3. Associate each transformed observation to one of the
        #    predicted (selected in Step 1) landmark positions.
        #    Use self.associate() for this purpose - it receives
        #    the predicted landmarks and observations; and returns
        #    the list of landmarks by implementing the nearest-neighbour
        #    association algorithm.
        # 4. Calculate probability of this set of observations based on
        #    a multi-variate Gaussian distribution (two variables being
        #    the x and y positions with means from associated positions
        #    and variances from std_landmark_x and std_landmark_y).
        #    The resulting probability is the product of probabilities
        #    for all the observations.
        # 5. Update the particle's weight by the calculated probability.

        for p in self.particles:
            landmarks = []
            for _id in map_landmarks:
                x, y = map_landmarks[_id]['x'], map_landmarks[_id]['y']
                if self.calculate_distance(x-p['x'], y-p['y']) < sensor_range:
                    landmarks.append({'id': _id, 'x': x, 'y': y})

            map_observations = []
            for observation in observations:
                x = p['x'] + observation['x']*np.cos(p['t']) - observation['y']*np.sin(p['t'])
                y = p['y'] + observation['x'] * np.sin(p['t']) + observation['y'] * np.cos(p['t'])
                map_observations.append({'x': x, 'y': y})

            if not landmarks:
                continue

            tmp_assoc = self.associate(landmarks, map_observations)

            p['w'] = 1.
            p['assoc'] = []
            for i in range(len(tmp_assoc)):
                p['w'] *= self.norm_pdf(self.calculate_distance(tmp_assoc[i]['x'] - p['x'], tmp_assoc[i]['y'] - p['y']),
                                        self.calculate_distance(observations[i]['x'], observations[i]['y']),
                                        math.sqrt(std_landmark_x**2 + std_landmark_y**2)) + 1e-100

                # p['w'] *= (self.norm_pdf(tmp[i]['x'] - p['x'], observations[i]['x'], std_landmark_x) *
                #            self.norm_pdf(tmp[i]['y'] - p['y'], observations[i]['y'], std_landmark_y)) + 1e-100

                # p['w'] *= self.multivariate_normal(np.array([observations[i]['x'],
                #                                         observations[i]['y']]),
                #                               2,
                #                               np.array([tmp[i]['x'] - p['x'],
                #                                         tmp[i]['y'] - p['y']]),
                #                               np.array([[1, 0.], [0., 1]])) + 1e-100
                p['assoc'].append(tmp_assoc[i]['id'])

    # Resample particles with replacement with probability proportional to
    #   their weights.
    def resample(self):
        import copy
        import random
        # TODO: Select (possibly with duplicates) the set of particles
        #       that captures the posteior belief distribution, by
        # 1. Drawing particle samples according to their weights.
        # 2. Make a copy of the particle; otherwise the duplicate particles
        #    will not behave independently from each other - they are
        #    references to mutable objects in Python.
        # Finally, self.particles shall contain the newly drawn set of
        #   particles.
        tmp = []
        _max = sum([i['w'] for i in self.particles])

        for _ in range(self.num_particles):
            r = random.uniform(0, _max)
            for particle in self.particles:
                if r < particle['w']:
                    tmp.append(copy.deepcopy(particle))
                    break
                else:
                    r -= particle['w']

        self.particles = tmp

    # Choose the particle with the highest weight (probability)
    def get_best_particle(self):
        highest_weight = -1.0
        for p in self.particles:
            if p['w'] > highest_weight:
                highest_weight = p['w']
                best_particle = p

        return best_particle

    def calculate_distance(self, x, y):
        return (x**2 + y**2)**(1/2)

    def norm_pdf(self, x, m, s):
        from math import sqrt, pi, exp
        one_over_sqrt_2pi = 1 / sqrt(2 * pi)
        return (one_over_sqrt_2pi / s) * exp(-0.5 * ((x - m) / s) ** 2)

    def multivariate_normal(self, x, d, mean, covariance):
        """pdf of the multivariate normal distribution."""
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
                np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))