
target = Funnel(d=20)


def full_b(X):
    X_sq = jnp.average(jnp.square(X), axis=0)

    def step(F2, index):
        x_sq = X_sq[index, -1]
        F2_new = (F2 * index + x_sq) / (index + 1)  # Update <f(x)> with a Kalman filter
        b = jnp.abs((F2_new - target.variance) / target.variance)

        return F2_new, b

    return jax.lax.scan(step, jnp.zeros(1), xs=jnp.arange(len(X_sq)))[1]



def funnel_ess(alpha, eps):


    sampler= Sampler(target, alpha*np.sqrt(target.d), eps, integrator= 'LF')

    X, n = sampler.sample(5000, 60)

    b2 = full_b(X)

    return 200.0/find_crossing(b2, 0.1)

#print(funnel_ess(1.0, 0.1))
#search_wrapper(funnel_ess, 0.5, 3.0, 0.05, 1.5, save_name='show')



df = pd.read_csv('submission/MCHMC/ICG/Table_ICG_MN_g.csv')
print(df['alpha'].tolist())
print(df['eps'].tolist())