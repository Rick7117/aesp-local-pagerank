import numpy as np
from numpy import bool_, sqrt, int64, float64
from numpy.linalg import norm
import math
import scipy.sparse as sp
import time
from numba import njit, objmode

@njit(cache=True)
def norm_inf(x):
    x = x.astype(np.float64)
    return np.max(np.abs(x))
@njit(cache=True)
def norm_one(x):
    x = x.astype(np.float64)
    return np.linalg.norm(x, 1)
@njit(cache=True)
def current_time():
    with objmode(t='float64'):
        t = time.perf_counter()
    return t



@njit(cache=True)
def appr(n, indptr, indices, degree, s, alpha, eps, opt_x):

    eps_vec = eps * degree
    const1 = .5 * (1. - alpha)

    # Tracking metrics
    runtime_acc = []
    errs = []
    opers = []
    grad_norms = []
    gammas = []
    
    debug_time = float64(0.)
    start_time = current_time()

    # Initial gradient setup
    r = s.copy().astype(np.float64)
    x = np.zeros_like(r)

    # Record initial metrics
    debug_start = current_time()
    runtime_acc.append(current_time()-start_time-debug_time)
    active_nodes = np.where(np.abs(r) >= eps_vec)[0]
    errs.append(norm_inf(x - opt_x) if opt_x is not None else 0.)
    opers.append(np.sum(degree[active_nodes]))
    curr_grad_norm = norm_one(r)
    grad_norms.append(curr_grad_norm)
    gammas.append(norm_one(r[active_nodes]) / curr_grad_norm)
    debug_time += current_time() - debug_start

    # Initialize queue
    front = rear = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    for u in np.arange(n):
        if eps_vec[u] <= r[u]:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
    rear = (rear + 1) % n
    queue[rear] = n 
    q_mark[n] = True

    # Main processing loop
    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n: 
            rear = (rear + 1) % n
            queue[rear] = n
            # Record metrics
            debug_start = current_time()
            runtime_acc.append(current_time()-start_time-debug_time)
            active_nodes = np.where(np.abs(r) >= eps_vec)[0]
            errs.append(norm_inf(x - opt_x) if opt_x is not None else 0.)
            opers.append(np.sum(degree[active_nodes]))
            curr_grad_norm = norm_one(r)
            grad_norms.append(curr_grad_norm)
            gammas.append(norm_one(r[active_nodes]) / curr_grad_norm)
            debug_time += current_time() - debug_start
            continue

        delta = const1 * r[u]
        x[u] += alpha * r[u]
        r[u] = delta
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= r[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    runtime = current_time() - start_time - debug_time
    return x, errs, opers, runtime, runtime_acc, grad_norms, None, gammas, eps

@njit(cache=True)
def apprOpt(n, indptr, indices, degree, s, alpha, eps, opt_x):

    sq_deg = sqrt(degree)
    eps_vec =  eps * sq_deg
    const1 = (1. - alpha) / (1. + alpha)
    const2 = (2. * alpha) / (1. + alpha)
    # Tracking metrics
    runtime_acc = []
    errs = []
    opers = []
    grad_norms = []
    gammas = []

    debug_time = float64(0.)
    start_time = current_time()

    # Initial gradient setup
    r = s.copy().astype(np.float64)
    x = np.zeros_like(r)

    # Record initial metrics
    debug_start = current_time()
    runtime_acc.append(current_time()-start_time-debug_time)
    active_nodes = np.where(np.abs(r) >= eps_vec)[0]
    errs.append(norm_inf(x - opt_x) if opt_x is not None else 0.)
    opers.append(np.sum(degree[active_nodes]))
    curr_grad_norm = norm_one(r)
    grad_norms.append(curr_grad_norm)
    gammas.append(norm_one(r[active_nodes]) / curr_grad_norm)
    debug_time += current_time() - debug_start

    # Initialize queue
    front = rear = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    for u in np.arange(n):
        if eps_vec[u] <= r[u]:
            rear = (rear + 1) % n
            queue[rear] = u
            q_mark[u] = True
    rear = (rear + 1) % n
    queue[rear] = n  # super epoch flag
    q_mark[n] = True

    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n:  # one local super-iteration
            rear = (rear + 1) % n
            queue[rear] = n
            
            # Record metrics
            debug_start = current_time()
            runtime_acc.append(current_time()-start_time-debug_time)
            active_nodes = np.where(np.abs(r) >= eps_vec)[0]
            errs.append(norm_inf(x - opt_x) if opt_x is not None else 0.)
            opers.append(np.sum(degree[active_nodes]))
            curr_grad_norm = norm_one(r)
            grad_norms.append(curr_grad_norm)
            gammas.append(norm_one(r[active_nodes]) / curr_grad_norm)
            debug_time += current_time() - debug_start
            continue
        
        delta = const1 * r[u]
        x[u] += const2 * r[u] 
        r[u] = 0.
        for v in indices[indptr[u]:indptr[u + 1]]:
            r[v] += delta / degree[u]
            if not q_mark[v] and eps_vec[v] <= r[v]:
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    runtime = current_time() - start_time - debug_time
    return x, errs, opers, runtime, runtime_acc, grad_norms, None, gammas, eps

@njit(cache=True)
def locGD(n, indptr, indices, degree, s, alpha, eps, opt_x):

    sq_deg = sqrt(degree)
    eps_vec = alpha * eps * sq_deg
    b0 = alpha * s / sq_deg
    const2 = (1 - alpha) / (1 + alpha)

    # Tracking metrics
    runtime_acc = []
    errs = []
    opers = []
    grad_norms = []
    gammas = []

    debug_time = float64(0.)
    start_time = current_time()

    # Initial gradient setup
    z = np.zeros(n, dtype=np.float64)
    grad_h = np.zeros(n, dtype=np.float64)
    grad_h[:] = -b0
    
    # Record initial metrics
    debug_start = current_time()
    runtime_acc.append(current_time()-start_time-debug_time)
    active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
    errs.append(norm_inf(z*sq_deg - opt_x))
    opers.append(np.sum(degree[active_nodes]))
    curr_grad_norm = norm_one(grad_h* sq_deg)
    grad_norms.append(curr_grad_norm)
    gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)
    debug_time += current_time() - debug_start

    # Initialize queue
    rear = int64(0)
    delta_st = np.nonzero(grad_h)[0]
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)

    while True:
        delta_vl = np.zeros(len(delta_st), dtype=float64)
        delta_vl[:] = grad_h[delta_st]
        z[delta_st] -= 2 / (1 + alpha) * delta_vl
        grad_h[delta_st] -= delta_vl

        for u, val in zip(delta_st, delta_vl):
            val = const2 * val / sq_deg[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad_h[v] += val / sq_deg[v]
                if not q_mark[v] and eps_vec[v] <= abs(grad_h[v]):
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        if rear == 0:
            break
        # updates for next
        delta_st = np.zeros(rear, dtype=int64)
        delta_st[:] = queue[:rear]
        q_mark[queue[:rear]] = False
        rear = 0

        debug_start = current_time()
        runtime_acc.append(current_time()-start_time-debug_time)
        active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
        errs.append(norm_inf(z*sq_deg - opt_x))
        opers.append(np.sum(degree[active_nodes]))
        curr_grad_norm = norm_one(grad_h* sq_deg)
        grad_norms.append(curr_grad_norm)
        gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)
        debug_time += current_time() - debug_start
    runtime = current_time() - start_time - debug_time
    return z*sq_deg, errs, opers, runtime, runtime_acc, grad_norms, None, gammas, eps

@njit(cache=True)
def _locAPPR(n, indptr, indices, degree, b, alpha, phi, eta, init_z, opt_x):

    sq_deg = sqrt(degree)
    m = len(indices) / 2.
    eps1 = sqrt((1. - alpha) * phi / m)

    # Tracking metrics
    errs = []
    opers = []
    grad_norms = []
    gammas = []


    
    # Initial gradient setup
    z = init_z.copy()
    grad_h = np.zeros(n, dtype=np.float64)
    grad_h[:] = -b
    for u in np.nonzero(z)[0]:
        grad_h[u] += (1. + alpha + 2 * eta) * z[u] / 2.
        for v in indices[indptr[u]:indptr[u + 1]]:
            grad_h[v] -= (1. - alpha) * z[u] / (2. * sq_deg[u] * sq_deg[v])
    
    eps2 = 2. * (1. - alpha) * phi / norm_one(sq_deg* grad_h)
    eps = max(eps1, eps2)
    eps_vec =  eps * sq_deg

    # Record initial metrics
    active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
    errs.append(norm_inf(z*sq_deg - opt_x))
    opers.append(np.sum(degree[active_nodes]))
    curr_grad_norm = norm_one(grad_h* sq_deg)
    grad_norms.append(curr_grad_norm)
    gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)

    debug_time = float64(0.)
    start_time = current_time()
    # Initialize queue
    front = rear = int64(0)
    queue = np.zeros(n + 1, dtype=int64)
    q_mark = np.zeros(n + 1, dtype=np.bool_)
    
    for u in active_nodes:
        rear = (rear + 1) % n
        queue[rear] = u
        q_mark[u] = True
    rear = (rear + 1) % n
    queue[rear] = n  
    q_mark[n] = True

    while (rear - front) != 1:
        front = (front + 1) % n
        u = queue[front]
        q_mark[u] = False
        if u == n: 
            rear = (rear + 1) % n
            queue[rear] = n
            # Record metrics
            debug_start = current_time()
            active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
            errs.append(norm_inf(z*sq_deg - opt_x))
            opers.append(np.sum(degree[active_nodes]))
            curr_grad_norm = norm_one(grad_h* sq_deg)
            grad_norms.append(curr_grad_norm)
            gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)
            active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
            debug_time += current_time() - debug_start
            continue
        
        delta = (1. - alpha) * grad_h[u] / (1. + alpha + 2. * eta)
        z[u] -= 2. * grad_h[u] / (1. + alpha + 2. * eta)
        grad_h[u] = 0.

        for v in indices[indptr[u]:indptr[u + 1]]:
            grad_h[v] += delta / (sq_deg[v] * sq_deg[u])
            if not q_mark[v] and eps_vec[v] <= abs(grad_h[v]):
                rear = (rear + 1) % n
                queue[rear] = v
                q_mark[v] = True
    runtime = current_time() - start_time - debug_time
    return z, errs, opers, runtime, grad_norms, gammas, eps

@njit(cache=True)
def aespAPPR(n, indptr, indices, degree, s, alpha, eps, opt_x):
    
    sq_deg = sqrt(degree)
    eps_vec = alpha * eps * sq_deg
    b0 = alpha * s / sq_deg

    eta = 1. - 2. * alpha
    m = np.sum(degree) / 2.
    beta = (sqrt(alpha + eta) - sqrt(alpha)) / (sqrt(alpha + eta) + sqrt(alpha))
    # inner-loop stop condition
    multi_phi = 1. - 0.9 * sqrt(alpha / (1 - alpha))
    # outer-loop iteration
    phi = (1+alpha) / 18
    T = 10 / 9 * sqrt((1 - alpha) / alpha) * np.log(400 * sqrt(1 - alpha ** 2) / (alpha**2 * eps ** 2))
    T = math.ceil(T)

    # Tracking metrics
    err_out = []
    oper_out = []
    runtime_inner = []
    grad_norm_all = []
    vol_all = []
    gamma_all = []
    eps_all = []

    # Initial gradient setup
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    grad_f = np.zeros(n, dtype=np.float64)
    grad_f[:] = -b0

    for t in range(1, T + 1):

        x_prev = x
        b = b0 + eta * y
        phi *= multi_phi

        x, errs, opers, runtime_in, grad_norms, gammas, eps_in = _locAPPR(n, indptr, indices, degree, b, alpha, phi, eta, y, opt_x)
        if opers == [0]:
            continue
        # Gradient Update
        grad_f[:] = -b0
        for u in np.nonzero(x)[0]:
            grad_f[u] += (1. + alpha) * x[u] / 2.
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad_f[v] -= (1. - alpha) * x[u] / (2. * sq_deg[u] * sq_deg[v])
        err_out.append(norm_inf(x*sq_deg - opt_x))
        oper_out.append(np.sum(np.array(opers, dtype=np.int64)))
        runtime_inner.append(runtime_in)
        grad_norm_all.append(grad_norms)
        vol_all.append(opers)
        gamma_all.append(gammas)
        eps_all.append(eps_in)

        if np.sum(np.abs(grad_f) >= eps_vec) == 0:
            break
        y = x + beta * (x - x_prev)
    runtime_inner_arr = np.array(runtime_inner, dtype=np.float64)
    return x*sq_deg, err_out, oper_out, np.sum(runtime_inner_arr), np.cumsum(runtime_inner_arr), grad_norm_all, vol_all, gamma_all, eps_all

@njit(cache=True)
def _locGD(n, indptr, indices, degree, b, alpha, phi, eta, init_z, opt_x):

    sq_deg = sqrt(degree)
    m = len(indices) / 2.
    const1 = 2 / (1 + alpha + 2 * eta)
    const2 = (1 - alpha) / (1 + alpha + 2 * eta)

    # Tracking metrics
    errs = []
    opers = []
    grad_norms = []
    gammas = []

    # Initial gradient setup
    z = init_z.copy()
    grad_h = np.zeros(n, dtype=np.float64)
    grad_h[:] = -b
    for u in np.nonzero(z)[0]:
        grad_h[u] += (1. + alpha + 2 * eta) * z[u] / 2.
        for v in indices[indptr[u]:indptr[u + 1]]:
            grad_h[v] -= (1. - alpha) * z[u] / (2. * sq_deg[u] * sq_deg[v])
    eps1 = sqrt((1. - alpha) * phi / m)
    eps2 = 2. * (1. - alpha) * phi / norm_one(sq_deg* grad_h)
    eps = max(eps1, eps2)
    eps_vec =  eps * sq_deg

    # Record initial metrics
    active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
    errs.append(norm_inf(z*sq_deg - opt_x))
    opers.append(np.sum(degree[active_nodes]))
    curr_grad_norm = norm_one(grad_h* sq_deg)
    grad_norms.append(curr_grad_norm)
    gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)
    
    debug_time = float64(0.)
    start_time = current_time()

    # Initialize queue
    rear = int64(0)
    delta_st = np.where(np.abs(grad_h) >= eps_vec)[0]
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)

    while True:
        delta_vl = np.zeros(len(delta_st), dtype=float64)
        delta_vl[:] = grad_h[delta_st]
        z[delta_st] -= const1 * delta_vl
        grad_h[delta_st] = 0.
        
        for u, val in zip(delta_st, delta_vl):
            val = const2 * val / sq_deg[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad_h[v] += val / sq_deg[v]
                if not q_mark[v] and eps_vec[v] <= abs(grad_h[v]):
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        if rear == 0:
            break
        # updates for next
        delta_st = np.zeros(rear, dtype=int64)
        delta_st[:] = queue[:rear]
        q_mark[queue[:rear]] = False
        rear = 0

        debug_start = current_time()
        active_nodes = np.where(np.abs(grad_h) >= eps_vec)[0]
        errs.append(norm_inf(z*sq_deg - opt_x))
        opers.append(np.sum(degree[active_nodes]))
        curr_grad_norm = norm_one(grad_h* sq_deg)
        grad_norms.append(curr_grad_norm)
        gammas.append(norm_one((sq_deg * grad_h)[active_nodes]) / curr_grad_norm)
        debug_time += current_time() - debug_start
    runtime = current_time() - start_time - debug_time
    return z, errs, opers, runtime, grad_norms, gammas, eps

@njit(cache=True)
def aespLocGD(n, indptr, indices, degree, s, alpha, eps, opt_x):
    
    sq_deg = sqrt(degree)
    eps_vec = alpha * eps * sq_deg
    b0 = alpha * s / sq_deg

    eta = 1. - 2. * alpha
    m = np.sum(degree) / 2.
    beta = (sqrt(alpha + eta) - sqrt(alpha)) / (sqrt(alpha + eta) + sqrt(alpha))
    # inner-loop stop condition
    multi_phi = 1. - 0.9 * sqrt(alpha / (1 - alpha))
    # outer-loop iteration
    phi = (1+alpha) / 18
    T = 10 / 9 * sqrt((1 - alpha) / alpha) * np.log(400 * sqrt(1 - alpha ** 2) / (alpha**2 * eps ** 2))
    T = math.ceil(T)

    # Tracking metrics
    err_out = []
    oper_out = []
    runtime_inner = []
    grad_norm_all = []
    vol_all = []
    gamma_all = []
    eps_all = []

    # Initial gradient setup
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    grad_f = np.zeros(n, dtype=np.float64)
    grad_f[:] = -b0

    for t in range(1, T + 1):

        x_prev = x
        b = b0 + eta * y
        phi *= multi_phi

        x, errs, opers, runtime_in, grad_norms, gammas, eps_in = _locGD(n, indptr, indices, degree, b, alpha, phi, eta, y, opt_x)
        if opers == [0]:
            continue
        # Gradient Update
        grad_f[:] = -b0
        for u in np.nonzero(x)[0]:
            grad_f[u] += (1. + alpha) * x[u] / 2.
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad_f[v] -= (1. - alpha) * x[u] / (2. * sq_deg[u] * sq_deg[v])

        err_out.append(norm_inf(x*sq_deg - opt_x))
        oper_out.append(np.sum(np.array(opers, dtype=np.int64)))
        runtime_inner.append(runtime_in)
        grad_norm_all.append(grad_norms)
        vol_all.append(opers)
        gamma_all.append(gammas)
        eps_all.append(eps_in)

        if np.sum(np.abs(grad_f) >= eps_vec) == 0:
            break
        y = x + beta * (x - x_prev)
    runtime_inner_arr = np.array(runtime_inner, dtype=np.float64)
    return x*sq_deg, err_out, oper_out, np.sum(runtime_inner_arr), np.cumsum(runtime_inner_arr), grad_norm_all, vol_all, gamma_all, eps_all

@njit(cache=True)
def ista(n, indptr, indices, degree, s, alpha, eps, rho_tilde, opt_x, l1_err):

    # initialize to avoid redundant calculation
    sq_deg = sqrt(degree)
    rho = rho_tilde / (1. + eps)
    b = 2*alpha/(1+alpha) * s / sq_deg
    eps_vec = rho * alpha * sq_deg

    # Tracking metrics
    errs = []
    opers = []
    runtime_acc = []
    grad_norms = []
    gammas = []

    # Initial gradient setup
    x = np.zeros(n, dtype=float64)
    delta_x = np.zeros(n, dtype=float64)
    grad_x = np.zeros(n, dtype=float64)

    debug_time = float64(0.)
    start_time = current_time()

    # Initialize queue
    queue = np.zeros(n, dtype=int64)
    q_mark = np.zeros(n, dtype=bool_)
    rear = int64(0)

    for u in np.nonzero(b)[0]:
        grad_x[u] = -(1. + alpha) * b[u] / 2.
        if (x[u] - grad_x[u]) >= eps_vec[u]:
            queue[rear] = u
            rear = rear + 1
            q_mark[u] = True

    while True:
        oper = 0
        st = queue[:rear]
        delta_x[st] = -(grad_x[st] + eps_vec[st])
        x[st] = (x[st] - grad_x[st]) - eps_vec[st]

        # --- debug ---
        debug_start = current_time()
        num = norm_one(grad_x[st[:rear]])
        dem = norm_one(grad_x)
        gammas.append(num / dem)
        debug_time += current_time() - debug_start
        # -------------

        for u in st:
            grad_x[u] += .5 * (1. + alpha) * delta_x[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sq_deg[v] * sq_deg[u]
                grad_x[v] -= .5 * (1. - alpha) * delta_x[u] / demon
                # new active nodes added into st
                if not q_mark[v] and (x[v] - grad_x[v]) >= eps_vec[v]:
                    queue[rear] = v
                    rear = rear + 1
                    q_mark[v] = True
            oper += degree[u]

        # ------ debug time ------
        debug_start = current_time()
        runtime_acc.append(current_time() - start_time - debug_time)
        errs.append(norm_inf((x*sq_deg - opt_x)/degree))
        opers.append(oper)
        grad_norms.append(norm_one(grad_x))
        debug_time += current_time() - debug_start
        # ------------------------

        st = queue[:rear]
        cond = np.max(np.abs(-grad_x[st] / degree[st]))
        if cond <= (1. + eps) * rho:
            break
        # minimal l1-err meets
        if l1_err is not None and errs[-1] <= l1_err:
            break
    runtime = current_time() - start_time - debug_time
    return x*sq_deg, errs, opers, runtime, runtime_acc, grad_norms, None, gammas, eps


@njit
def cheby(n, indptr, indices, degree, s_node, alpha, eps, opt_x):
    with objmode(start='f8'):
        start = time.perf_counter()
    # ----------------------
    xt_tilde = np.zeros(n, dtype=float64)
    xt = np.zeros(n, dtype=float64)
    rt = np.zeros(n, dtype=float64)
    # ----------------------
    # queue data structure
    sq_deg = sqrt(degree)
    b = 2*alpha/(1+alpha) * s_node / sq_deg
    s = np.nonzero(b)[0]
    rt[s[0]] = b[s[0]] * np.sqrt(degree[s[0]])
    xt_tilde[s[0]] = rt[s[0]]
    xt[s[0]] = rt[s[0]]
    queue = np.zeros(n, dtype=int64)
    queue[:len(s)] = s
    q_mark = np.zeros(n, dtype=bool_)
    q_mark[s] = True
    rear = len(s)
    eps_vec = eps * degree
    const = (1. - alpha) / (1. + alpha)
    delta_t = const

    st1 = np.zeros(n, dtype=int64)
    vl1 = np.zeros(n, dtype=float64)
    st1_len = rear
    st1[:st1_len] = queue[:st1_len]
    vl1[:st1_len] = rt[st1[:st1_len]]
    q_mark[st1[:st1_len]] = False

    rear = 0
    rt[st1[:st1_len]] = 0.
    for ind in range(st1_len):
        u = st1[ind]
        val = const * vl1[ind] / degree[u]
        for v in indices[indptr[u]:indptr[u + 1]]:
            rt[v] += val
            if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                queue[rear] = v
                q_mark[v] = True
                rear += 1
    st2 = np.zeros(n, dtype=int64)
    vl2 = np.zeros(n, dtype=float64)
    st2_len = rear
    # ----------------------
    errs = [norm_inf(opt_x/degree)]
    opers = [0.]
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)

    while True:

        delta_t = 1. / (2. / const - delta_t)
        beta = 2. * delta_t / const

        # updates for current iteration from queue
        if st2_len < n / 4:
            st2[:st2_len] = queue[:st2_len]
        else:  # continuous memory
            st2[:st2_len] = np.nonzero(q_mark)[0]
        # st2[:st2_len] = queue[:st2_len]

        vl2[:st2_len] = beta * rt[st2[:st2_len]] + (beta - 1.) * xt_tilde[st2[:st2_len]]
        q_mark[st2[:st2_len]] = False

        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        num = np.linalg.norm(rt[st2[:st2_len]] / sq_deg[st2[:st2_len]], 1)
        dem = np.linalg.norm(rt / sq_deg, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------

        rear = 0
        xt[st2[:st2_len]] += vl2[:st2_len]
        rt[st2[:st2_len]] -= vl2[:st2_len]
        for ind in range(st2_len):
            u = st2[ind]
            val = const * vl2[ind] / degree[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                rt[v] += val
                if not q_mark[v] and eps_vec[v] <= np.abs(rt[v]):
                    queue[rear] = v
                    q_mark[v] = True
                    rear += 1
        xt_tilde[st2[:st2_len]] += vl2[:st2_len]
        xt_tilde[st1[:st1_len]] -= vl1[:st1_len]

        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        # minimal l1-err meets
        if opt_x is not None:
            err = norm_inf((xt*sq_deg - opt_x)/degree)
            errs.append(err)
        opers.append(np.sum(degree[st1[:st1_len]]))
        cd_xt.append(np.count_nonzero(xt))
        cd_rt.append(np.count_nonzero(rt))
        vol_st.append(np.sum(degree[np.nonzero(rt)]))
        vol_it.append(np.sum(degree[np.nonzero(rt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        st1[:st2_len] = st2[:st2_len]
        vl1[:st2_len] = vl2[:st2_len]
        st1_len = st2_len
        st2_len = rear

        # queue is empty now, quit
        if rear == 0:
            break

    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return xt, errs, opers, run_time, op_time, rt, vol_st, gamma_t, eps

@njit
def fista(n, indptr, indices, degree, s, alpha, eps, rho_tilde, mome_fixed, opt_x, l1_err):
    with objmode(start='f8'):
        start = time.perf_counter()
    # queue to maintain active nodes per-epoch
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    q_len = np.int64(0)
    # approximated solution
    qt = np.zeros(n, dtype=np.float64)
    yt = np.zeros(n, dtype=np.float64)
    grad_yt = np.zeros(n, dtype=np.float64)
    # initialize to avoid redundant calculation
    b = 2*alpha/(1+alpha) * s / sqrt(degree)
    sq_deg = sqrt(degree)
    rho = rho_tilde / (1. + eps)
    eps_vec = rho * alpha * sq_deg
    for u in np.nonzero(b)[0]:
        grad_yt[u] = -(1. + alpha) * b[u] / 2.
        if (qt[u] - grad_yt[u]) >= eps_vec[u]:
            queue[q_len] = u
            q_len += 1
            q_mark[u] = True

    # results
    errs = [norm_inf((yt*sq_deg - opt_x)/degree)]
    opers = [0.]
    cd_xt = []
    cd_rt = []
    vol_st = []
    vol_it = []
    gamma_t = []
    op_time = np.float64(0.)
    # parameter for momentum
    t1 = 1
    beta = (1. - np.sqrt(alpha)) / (1. + np.sqrt(alpha))
    while True:
        for ind in range(q_len):
            q_mark[queue[ind]] = False
        rear = 0
        oper = 0.
        # --- debug ---
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        num = np.linalg.norm(grad_yt[queue[:q_len]], 1)
        dem = np.linalg.norm(grad_yt, 1)
        gamma_t.append(num / dem)
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # -------------
        for ind in range(q_len):
            u = queue[ind]
            if (yt[u] - grad_yt[u]) >= eps_vec[u]:
                delta_qi = yt[u] - grad_yt[u] - eps_vec[u] - qt[u]
            elif (yt[u] - grad_yt[u]) <= - eps_vec[u]:
                delta_qi = yt[u] - grad_yt[u] + eps_vec[u] - qt[u]
            else:
                delta_qi = -qt[u]
            qt[u] += delta_qi
            if mome_fixed:
                delta_yi = qt[u] + beta * delta_qi - yt[u]
            else:
                t_next = .5 * (1. + np.sqrt(4. + t1 ** 2.))
                beta = (t1 - 1.) / t_next
                delta_yi = qt[u] + beta * delta_qi - yt[u]
                t1 = t_next

            yt[u] += delta_yi
            grad_yt[u] += .5 * (1. + alpha) * delta_yi
            for j in indices[indptr[u]:indptr[u + 1]]:
                demon = sq_deg[j] * sq_deg[u]
                ratio = .5 * (1 - alpha) / demon
                grad_yt[j] += (- ratio * delta_yi)
                if not q_mark[j] and np.abs(grad_yt[j]) > eps_vec[j] * (1. + eps):
                    queue[rear] = j
                    rear += 1
                    q_mark[j] = True
            oper += degree[u]
        # ------ debug time ------
        with objmode(debug_start='f8'):
            debug_start = time.perf_counter()
        if opt_x is not None:
            errs.append(norm_inf((yt*sq_deg - opt_x)/degree))
        opers.append(oper)
        cd_xt.append(np.count_nonzero(yt))
        cd_rt.append(np.count_nonzero(grad_yt))
        vol_st.append(oper)
        vol_it.append(np.sum(degree[np.nonzero(grad_yt)]))
        with objmode(op_time='f8'):
            op_time += (time.perf_counter() - debug_start)
        # ------------------------
        q_len = rear
        if q_len == 0:
            break
        # minimal l1-err meets
        if l1_err is not None and errs[-1] <= l1_err:
            break
    with objmode(run_time='f8'):
        run_time = time.perf_counter() - start
    return yt*sq_deg, errs, opers, run_time, op_time, grad_yt, vol_st, gamma_t, eps

@njit
def __apgd(indptr, indices, sqrt_deg, alpha, queue, rear, x0, t, s, rho, delta_t):
    at_pre = 0.
    at = 1.
    yt = np.copy(x0)
    zt = np.copy(x0)
    kappa = 1. / alpha
    st = queue[:rear]
    num_oper = 0
    eps_vec = alpha * delta_t
    for _ in np.arange(t):
        at_next = at_pre + at
        xt = (at_pre / at_next) * yt + (at / at_next) * zt
        coeff_1 = (kappa - 1. + at_pre) / (kappa - 1. + at_next)
        coeff_2 = at / (kappa - 1. + at_next)
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon
            num_oper += sqrt_deg[u] ** 2.
        tmp_zt = coeff_1 * zt + coeff_2 * (xt - grad_xt / alpha)
        zt = np.zeros(len(xt))
        for u in st:
            if tmp_zt[u] > 0:
                zt[u] = tmp_zt[u]
        yt = (at_pre / at_next) * yt + (at / at_next) * zt
        at = at_next * (2. * kappa / (2. * kappa + 1. - np.sqrt(1. + 4 * kappa)) - 1.)
        at_pre = at_next
        if not np.any(np.abs(grad_xt)>= eps_vec):
            break   
    return yt, num_oper

@njit(cache=True)
def aspr(n, indptr, indices, degree, s, alpha, eps, rho, opt_x):
    xt = np.zeros(n, dtype=np.float64)
    queue = np.zeros(n, dtype=np.int64)
    q_mark = np.zeros(n, dtype=np.bool_)
    sqrt_deg = np.sqrt(degree)
    rear = 0
    new_counts = 0
    for i in np.arange(n):
        if s[i] > rho * degree[i]:
            queue[rear] = i
            q_mark[i] = True
            rear += 1
            new_counts += 1
    l1_error = []
    nonzero_list = []
    st_list = []
    num_opers = []
    num_oper = 0.
    eps_vec = alpha * eps * sqrt_deg
    while new_counts != 0:
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        st = queue[:rear]

        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon
            num_oper += degree[u]

        delta_t = np.sqrt((eps * alpha) / (1. + rear))
        eps_t_hat = (alpha * (delta_t ** 2.)) / 2.
        num = (1. - alpha) * np.sum(grad_xt[st] ** 2.)
        dem = 2. * eps_t_hat * (alpha ** 2.)
        t = 1. + np.ceil(2. * np.sqrt(1. / alpha) * np.log(num / dem))
        xt_bar, num_oper_ = __apgd(
            indptr, indices, sqrt_deg, alpha, queue, rear, xt, t, s, rho, delta_t)
        num_oper += num_oper_
        for u in st:
            if (xt_bar[u] - delta_t) > 0.:
                xt[u] = xt_bar[u] - delta_t
            else:
                xt[u] = 0.
        # calculate the gradient
        grad_xt = alpha * (rho * sqrt_deg - s / sqrt_deg)
        for u in st:
            grad_xt[u] += .5 * (1. + alpha) * xt[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                demon = sqrt_deg[v] * sqrt_deg[u]
                grad_xt[v] -= .5 * (1. - alpha) * xt[u] / demon

        st_count_old = rear
        for i in np.arange(n):
            if grad_xt[i] < 0. and not q_mark[i]:
                queue[rear] = i
                rear += 1
                q_mark[i] = True
        new_counts = rear - st_count_old
        if opt_x is not None:
            nonzero_list.append(np.count_nonzero(xt))
            l1_error.append(norm_inf((xt*sqrt_deg - opt_x)/degree))
            st_list.append(rear)
            num_opers.append(num_oper)
    return xt*sqrt_deg, l1_error, num_opers, None, None, None, None, None, None

@njit(cache=True)
def aespAPPR_init(n, indptr, indices, degree, s, alpha, eps, opt_x, init):
    
    sq_deg = sqrt(degree)
    eps_vec = alpha * eps * sq_deg
    b0 = alpha * s / sq_deg

    eta = 1. - 2. * alpha
    m = np.sum(degree) / 2.
    beta = (sqrt(alpha + eta) - sqrt(alpha)) / (sqrt(alpha + eta) + sqrt(alpha))
    # inner-loop stop condition
    multi_phi = 1. - 0.9 * sqrt(alpha / (1 - alpha))
    # outer-loop iteration
    phi = (1+alpha) / 18
    T = 10 / 9 * sqrt((1 - alpha) / alpha) * np.log(400 * sqrt(1 - alpha ** 2) / (alpha**2 * eps ** 2))
    T = math.ceil(T)

    # Tracking metrics
    err_out = []
    oper_out = []
    runtime_inner = []
    grad_norm_all = []
    vol_all = []
    gamma_all = []
    eps_all = []

    # Initial gradient setup
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    zero_init = np.zeros(n, dtype=np.float64)
    grad_f = np.zeros(n, dtype=np.float64)
    grad_f[:] = -b0

    for t in range(1, T + 1):

        x_prev = x
        b = b0 + eta * y
        phi *= multi_phi

        if init == 'y':
            x, errs, opers, runtime_in, grad_norms, gammas, eps_in = _locAPPR(n, indptr, indices, degree, b, alpha, phi, eta, y, opt_x)
        elif init == 'x':
            x, errs, opers, runtime_in, grad_norms, gammas, eps_in = _locAPPR(n, indptr, indices, degree, b, alpha, phi, eta, x_prev, opt_x)
        else:
            x, errs, opers, runtime_in, grad_norms, gammas, eps_in = _locAPPR(n, indptr, indices, degree, b, alpha, phi, eta, zero_init, opt_x)
        
        if opers == [0]:
            continue
        # Gradient Update
        grad_f[:] = -b0
        for u in np.nonzero(x)[0]:
            grad_f[u] += (1. + alpha) * x[u] / 2.
            for v in indices[indptr[u]:indptr[u + 1]]:
                grad_f[v] -= (1. - alpha) * x[u] / (2. * sq_deg[u] * sq_deg[v])
        err_out.append(norm_inf(x*sq_deg - opt_x))
        oper_out.append(np.sum(np.array(opers, dtype=np.int64)))
        runtime_inner.append(runtime_in)
        grad_norm_all.append(grad_norms)
        vol_all.append(opers)
        gamma_all.append(gammas)
        eps_all.append(eps_in)

        if np.sum(np.abs(grad_f) >= eps_vec) == 0:
            break
        y = x + beta * (x - x_prev)
    runtime_inner_arr = np.array(runtime_inner, dtype=np.float64)
    return x*sq_deg, err_out, oper_out, np.sum(runtime_inner_arr), np.cumsum(runtime_inner_arr), grad_norm_all, vol_all, gamma_all, eps_all


# if __name__ == "__main__":
    graph_path = '/mnt/data/binbin/git/ICML_2025_code_review/datasets/com-dblp/com-dblp_csr-mat.npz'
    adj_m = sp.load_npz(graph_path)
    degree = adj_m.sum(1).A1.astype(np.int64)
    sq_deg = np.sqrt(degree)
    indices = adj_m.indices
    indptr = adj_m.indptr
    n = adj_m.shape[0]
    alpha = 1e-1
    eps = 1e-6
    seed_node = 1
    s = np.zeros(n)
    s[seed_node] = 1.
    eta = 0
    b = 2*alpha/(1+alpha) * s / sq_deg

    re_ = appr(n, indptr, indices, degree, s, alpha, 1e-1/n, opt_x=None)
    opt_x = re_[0]
    print(norm_one(opt_x))
    # re_ = fista(n, indptr, indices, degree, s, alpha, eps, eps/1.1, False, opt_x, l1_err=None)
    # re_ = ista(n, indptr, indices, degree, s, alpha, eps, eps/1.1, opt_x, l1_err=None)
    re_ = cheby(n, indptr, indices, degree, s, alpha, eps, opt_x)
    print(norm_one(re_[0]))
    print(re_[1])