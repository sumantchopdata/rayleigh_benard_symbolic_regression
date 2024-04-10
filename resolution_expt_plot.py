#%%
# plot the results from reading a .txt file
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('resolution_expt_results.txt', header=None)

# create a plot of the data as circles with first and second columns on the axes,
# where the size of the circle is proportional to the value in the third column

fig, ax = plt.subplots()
plt.grid(visible=True)
ax.scatter(df[0], df[1], s=500*df[2])
ax.set_xlabel('The number of steps in X direction')
ax.set_ylabel('The number of steps in Z direction')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

# also plot the values as text
for i, txt in enumerate(df[2]):
    ax.annotate(round(txt,3), (df[0][i], df[1][i]))

plt.title('The effect of resolution on R2 value\n\
The area of the circle is proportional to R2')
plt.savefig('resolution_expt_results.png')
plt.show()