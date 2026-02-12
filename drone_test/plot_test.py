import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import time

num_points = 10
x = np.linspace(1, 10, num_points)
y = np.linspace(1, 10, num_points)
z = np.linspace(1, 10, num_points)
points = np.array([x, y, z])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Trajectory')
ax.view_init(elev=80, azim=-90)

realtime_marker, = ax.plot(1, 2, 3, 'ro')
realtime_line, = ax.plot([], [], [], 'b--')
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_zlim(0, 11)

plt.ion()
plt.show(block=False)

for i in range (10):
    realtime_marker.set_data([points[0,i]], [points[1,i]])
    realtime_marker.set_3d_properties([points[2, i]])
    realtime_line.set_data(points[0,:i+1], points[1,:i+1])
    realtime_line.set_3d_properties([points[2,:i+1]])
    fig.canvas.draw_idle()      
    fig.canvas.flush_events()  
    time.sleep(0.5)

input("Press Enter to close the plot...")