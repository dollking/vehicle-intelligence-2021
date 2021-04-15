# Week 4 - Motion Model & Particle Filters

---

[//]: # (Image References)
[empty-update]: ./empty-update.gif
[example]: ./example.gif
[result1]: ./particle_result_1.gif
[result2]: ./particle_result_2.gif
[result3]: ./particle_result_3.gif

## Assignment

You will complete the implementation of a simple particle filter by writing the following two methods of `class ParticleFilter` defined in `particle_filter.py`:

* `update_weights()`: For each particle in the sample set, calculate the probability of the set of observations based on a multi-variate Gaussian distribution.
* `resample()`: Reconstruct the set of particles that capture the posterior belief distribution by drawing samples according to the weights.

To run the program (which generates a 2D plot), execute the following command:

```
$ python run.py
```

Without any modification to the code, you will see a resulting plot like the one below:

![Particle Filter without Proper Update & Resample][empty-update]

while a reasonable implementation of the above mentioned methods (assignments) will give you something like

![Particle Filter Example][example]

Carefully read comments in the two method bodies and write Python code that does the job.

## Assignment Result
### Default Parameter Result
![Default parameter][result1]

### Landmark std 10 times Result
![LD std 3.0][result2]

### Position std 10 times Result
![Pos std 3.0][result3]