import numpy as np

def gen_traj(model, init, T_i, T_o, T):
    traj = init.reshape(T_i, 2).tolist()    
    for i in range(T):
        try:
            seg = model.predict([np.array(traj[i:i+T_o]).flatten()])
        except:
            seg = model.predict(np.array(traj[i:i+T_o]).reshape(1,-1))
        x = seg.reshape(T_i, 2).tolist()[0]
        traj.append(x)
    return np.array(traj)

def unzero_traj(traj, i, class_id, is_train=True):
    # find original id from train_test_ids
    id = train_test_ids[class_id]['train'][i] if is_train else train_test_ids[class_id]['test'][i]

    # get the first value from raw traj
    zero = all_trajs_raw[class_id][id][0]

    # reconstruct new raw traj
    unzeroed_traj = traj + zero
    
    return unzeroed_traj