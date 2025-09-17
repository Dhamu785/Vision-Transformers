# %% imports
import torch as t
# %% Rolling definition 
a = t.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])
print("Shape = ", a.shape)
print("Array = ",a)
# %% Rolling implementation
rolled = t.roll(a, shifts=(2,2), dims=(0,1))
print(rolled)
# %%
