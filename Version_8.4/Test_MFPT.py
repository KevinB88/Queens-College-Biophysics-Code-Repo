
# testing a computation for the mean passage flow time
def compute_mfpt(mass_loss_container, d_time):
    mfpt = 0
    for k in range(len(mass_loss_container)):
        mfpt += mass_loss_container[k] * (k * d_time) * d_time
    return mfpt
