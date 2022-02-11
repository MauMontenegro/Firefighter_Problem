from ete3 import Tree
import numpy as np

t = Tree("((2-0,2-2,0-2)1-1)0-0;", format=8)
print(t.write())
print(t)
# To print the internal node names you need to change the format:
print(t.write(format=1))
print(t)
# We can also write into a file
t.write(format=1, outfile="new_tree.nw")

# Now that we have the Burning Tree, we need to search for each feasible node the total of nodes that we are saving.
    cutting_node = '4-4 '
    saved = SavedNodes(t, cutting_node)

    # Random Policy for at most "threshold" steps
    while not done and step < threshold:
        fig = env.render(mode="rgb_array")
        fig.savefig('Images/Emulation_{f}.png'.format(f=step))
        plt.close()
        action = env.action_space.sample()  # Your agent goes here!
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    print(f"{ProtoEnv}")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward}")