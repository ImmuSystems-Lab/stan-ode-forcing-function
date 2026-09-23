#!/usr/bin/env python
# coding: utf-8

# # Using forcing functions in ordinary differential equations (ODEs) in Stan
# 
# ## Overview
# 
# Ordinary differential equations (ODEs)
# are widely used for mathematical modeling
# of natural systems.
# A less common
# but critical
# ODE solver input
# is the incorporation of fixed temporal data
# representing a process whose underlying equation is unknown,
# called a forcing function.
# Forcing functions are more readily supported
# in lower-level programming languages
# than in higher-level, domain-specific programming languages (DSLs)
# like Stan
# that allow fewer mathematical primitives
# and are more complex to extend.
# The sections below
# explain
# how to use a forcing function
# with the widely used 2-equation predator–prey ODEs example,
# also called Lotka–Volterra.
# 
# ## Preamble
# 
# First, load the modules we will need for the entire exercise.

# In[1]:


# Python builtin modules.
from argparse import Namespace
from collections import OrderedDict
import re

# 3rd party modules.
import arviz as az
import arviz_plots as azp
import cmdstanpy
from lets_plot import (
    LetsPlot, aes, as_discrete, facet_wrap, ggplot, geom_line, geom_point,
    geom_vline, labs, scale_y_log10,
)
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.integrate import solve_ivp
import scipy.stats
import statsmodels.formula.api as smf

# Configuration.
az.rcParams["plot.backend"] = "matplotlib"
LetsPlot.setup_html()


# In[2]:


get_ipython().run_line_magic('load_ext', 'cmdstanjupyter')


# In[3]:


# One time Stan setup.
assert cmdstanpy.install_cmdstan()


# The data science library `pandas`
# requires little introduction for many Python users.
# 
# > **Note:**
# > Those less familiar
# > can learn about them
# > from the freely available Carpentries lesson
# > *Programming with Python*
# > (<https://swcarpentry.github.io/python-novice-inflammation/>)
# > and, for a more depth, the excellent freely available book
# > *Python for Data Analysis*
# > (<https://wesmckinney.com/book/>).
# 
# > **Note:**
# > The grammar of graphics implemented by `lets_plot`
# > is probably more well known among R users,
# > so users coming from matplotlib, seaborn, etc.
# > may benefit from the freely available book *Python for Data Science*
# > that provides a nice overview of the lets-plot Python library
# > (<https://aeturrell.github.io/python4DS/data-visualise.html>).
# 
# While we our primarily goal is to use `stan` to solve the ODEs,
# we use `scipy.integrate.solve_ivp` to generate the *ground truth* data;
# ground truth data here refers to
# generating observations with known *true* parameters
# so that we can assess how well
# the fitted parameters
# compare with the true parameters.
# 
# > **Note:**
# > It is also possible to use Stan (`stan`) itself
# > to generate the ground truth data,
# > as described in the
# > Stan User's Guide >
# > Example Models >
# > Ordinary Differential Equations >
# > Measurement error models.
# > But for simplicity,
# > we use Python's popular `solve_ivp` function
# > (that replaces the deprecated `odeint` function)
# > to use familiar Python code where possible.
# 
# A well written Stan model
# should be relatively insensitive to initial conditions.
# The example below,
# demostrates a case of undesirable sensitivity to initial conditions.
# ***FIXME:*** The `posterior` package
# simplifies inspecting the data wrangling work
# to visualize the ODE parameters accepted by Stan
# for each chain to troubleshoot disagreement among chains.
# 
# ## Typical ODEs without a forcing function
# 
# ### Equations in Python
# 
# $$
# \begin{aligned}
# \frac{dx}{dt} &= \phantom{-} \alpha x - \beta x y \\
# \frac{dy}{dt} &= - \gamma y + \delta x y \\
# \end{aligned}
# $$
# where,
# 
# State variables:
# $$
# \begin{aligned}
# x &: \text{Population of prey} \\
# y &: \text{Population of predators} \\
# \end{aligned}
# $$
# Parameters:
# $$
# \begin{aligned}
# \alpha &: \text{Growth rate of prey} \\
# \beta &: \text{Death rate of prey due to predators} \\
# \gamma &: \text{Death rate of predators} \\
# \delta &: \text{Growth rate of predators due to prey}
# \end{aligned}
# $$
# 
# To code these equations in Python,
# we need to wrap them in a function
# with specific function arguments
# common to many ODE solvers,
# namely,
# the time vector,
# the list of state variables,
# and then the list of parameters.
# Therefore,
# the input to `solve_ivp()` from the `scipy` package
# is written as:

# In[4]:


true_y0 = OrderedDict((
    ("x", 10,),
    ("y", 5,),
))
true_params = OrderedDict((
    ("a", 1.0,),
    ("b", 0.5,),
    ("g", 1.5,),
    ("d", 0.7,),
))


def lv(t, y_, p=true_params):
    """Right-hand side of the Lotka–Volterra ODEs.

    Parameters
    ----------
    t : float
        Single timepoint at which to solve the system of equations.
    y_ : array_like
        State variables at the timepoint `t`.
    p : dict of float, optional
        Model parameters.

    """
    # Unpack y into convenience variances that must match the return value of
    # this function.
    x, y = y_

    # Decode the initial parameters and initial state variable values.
    p = Namespace(**p)

    # Encode the equation terms.
    prey_growth = p.a * x
    prey_death = p.b * x * y
    predator_growth = p.d * x * y
    predator_death = p.g * y

    # Return the equation RHS.
    dx_dt = prey_growth - prey_death
    dy_dt = predator_growth - predator_death
    return (dx_dt, dy_dt,)


# The parameters are stored in an `OrderedDict` object
# to improve legibility
# by matching the equation order of parameters.
# The `Namespace` unpacks the parameters
# more conveniently with dot notation (`p.alpha`)
# instead of dictionary square brackets (`p["alpha"]`).
# However,
# in production code,
# some of these niceties may need to be omitted,
# especially if they significantly effect the ODE solver speed
# as benchmarked by Jupyter's `%timeit`, etc.
# 
# ### Generate noisy input data with known ODE parameters
# 
# A common statistical practice
# is to generate data with known parameters
# so that we can test
# how well a method or model
# is able to recover the parameters used to generate the data.
# Therefore,
# we next simulate ODE data
# using known parameters
# `a`, `b`, `g`, and `d`
# (i.e. $\alpha$, $\beta$, $\gamma$, and $\delta$).
# Additionally,
# we need to assume a known initial condition
# $x_0$ and $y_0$
# stored in `true_y0`.
# Lastly,
# to make the parameter fitting
# more challenging and realistic to real-world observations,
# we add simulated noise.
# 

# In[5]:


# Solve the ODEs, then add noise to each variable using the Normal
# distribution.
times = np.linspace(0, 15, 31)
pristine = solve_ivp(fun=lv,
                     t_span=[np.min(times), np.max(times)],
                     y0=list(true_y0.values()),
                     t_eval=times)
# Check there was no solver error.
assert pristine.success
# Add noise to the pristine ODE solution.
noisy = pristine.y.copy()
prng = np.random.Generator(np.random.MT19937(seed=123))
noisy += prng.normal(size=noisy.shape, scale=0.5)
noisy[noisy < 0] = 0


# Using `solve_ivp()` above only solves the system of equations
# at the timepoints specified by `times`.
# The simulated "pristine" and "noisy" data points are shown below,
# although from here on we will only use the noisy simulated data.

# In[6]:


def ode_to_df(arr, time=pristine.t[:, np.newaxis]):
    return pd.DataFrame(np.concat([time, arr.T], axis=1),
                        columns=["time"] + list(true_y0.keys())[:arr.shape[0]])


data = (
    (
        pd.concat([ode_to_df(pristine.y), ode_to_df(noisy)],
                  keys=["pristine", "noisy"])
        .reset_index(level=0, names=["type"])
    )
        .reset_index(drop=True)
).reset_index(names="i")
data.groupby("type").head()


# In[7]:


data = pd.melt(data, id_vars=["i", "type", "time"])
data


# In[8]:


(
    ggplot(data,
           aes("time", "value", color="variable")) +
    facet_wrap("type", ncol=1) +
    geom_line() +
    geom_point()
)


# ### Fit the ODE parameters using the equivalent Stan model
# 
# The model we are going to fit
# is saved in a separate `lv.stan` file
# that looks like this:

# In[9]:


get_ipython().run_line_magic('stanf', 'lv.stan model_lv')


# In[10]:


fit1 = model_lv.sample(chains=4,
                       parallel_chains=4,
                       threads_per_chain=1,
                       iter_warmup=1_000,
                       iter_sampling=5_000,
                       seed=123,
                       show_progress=False,
                       data={
    "T": noisy.T.shape[0] - 1,
    "y": noisy.T[1:,],
    "y0": list(true_y0.values()),
    "t0": 0,
    "ts": pristine.t[1:],
    "g": true_params["g"],
    "d": true_params["d"],
})


# In[11]:


print(fit1.diagnose())


# We get warnings
# about how our initial conditions
# were not able to agree
# and mix together
# that we will explore in the next section.
# 
# ### Inspect the Stan fit

# In[12]:


## Numerical summary of the fits.
fit1.summary()


# In[13]:


## Prepare data for plotting.
idata1 = az.from_cmdstanpy(fit1)


# In[14]:


## Parameter plot of the fits.
azp.plot_forest(idata1)


# In[15]:


## Show all samples after warmup.
azp.plot_trace(idata1)


# In[16]:


## Show the pairs plot.
## FIXME: Possible to include lp__?
azp.plot_pair(idata1,
              visuals={
                  "contour": {"color": "black", "alpha": 0.3},
                  "scatter": {"size": 2, "alpha": 0.1},
             })


# In[17]:


## Plot the ODEs using a sample of 100 parameters from each of the chains.
df_fit1 = fit1.draws_pd()
## Sample greater variation by only choosing one sample from each qcut.
qcut_20 = lambda x: pd.qcut(x, np.linspace(0, 1, 20), labels=False)
grouped = (
    df_fit1
    .filter(["chain__", "iter__", "a", "b"])
    .groupby("chain__")
)
n = 100
prng = np.random.Generator(np.random.MT19937(seed=123))
df_fit1_subset = (
    df_fit1.join(
        pd.DataFrame({
            "qa": grouped["a"].transform(qcut_20),
            "qb": grouped["b"].transform(qcut_20),
        })
    )
    .groupby(["chain__", "qa", "qb"])
    .sample(n=1, random_state=prng)
    ## Now sample the 100 parameters from each chain.
    .groupby("chain__")
    .sample(n=n, replace=True)
    .reset_index()
    .reset_index(names="sample")
)
n_ode = df_fit1_subset.shape[0]
arr = np.empty(np.array(pristine.y.T.shape) * np.array([n_ode, 1]),
               dtype=pristine.y.dtype)
for i in range(n_ode):
    ode = solve_ivp(fun=lv,
                    t_span=[np.min(times), np.max(times)],
                    y0=noisy.T[0, :],
                    t_eval=times,
                    args=[
        {
            "a": df_fit1_subset.loc[i, "a"],
            "b": df_fit1_subset.loc[i, "b"],
            "g": true_params["g"],
            "d": true_params["d"],
        }
    ])
    assert ode.success
    arr[(i * len(times)):((i + 1) * len(times)), ] = ode.y.T
data = (
    pd.concat(
        (
            pd.DataFrame(
                {
                    "sample": np.repeat(range(n_ode), len(times)),
                    "time": np.tile(times, n_ode),
                }
            ),
            pd.DataFrame(arr, columns=true_y0.keys())
        ),
        axis=1
    )
    .join(df_fit1_subset.filter(["sample", "chain__"]),
          on="sample",
          lsuffix="l",
          rsuffix="r")
)
data = pd.melt(data, id_vars=["sample", "time", "chain__", "samplel", "sampler"])
data


# In[18]:


(
    ggplot(data, aes("time", "value", group=["sample", "variable"], color="variable")) +
    facet_wrap("chain__") +
    geom_line()
)


# In the last plot above
# that shows 100 ODE samples simulated by each of the 4 chains,
# we see that three of the chains
# are not able to capture the true periodicity.
# This is because of two reasons:
# (1) the oscillatory ODE system contains "peaks"
# of higher posterior probabilities
# and moving between these "peaks" is hard
# and (2) we are using large priors
# `a ~ normal(0.5, 0.5)` and `b ~ normal(0.5, 0.5)`:

# In[19]:


## Plot the priors for `a` and `b` used in the Stan model.
data = (
    pd.DataFrame({"x": np.linspace(0, 10, 101)})
    .assign(y=lambda df: scipy.stats.norm(loc=0.5, scale=0.5).pdf(df["x"]))
)
(
    ggplot(data, aes("x", "y")) +
    geom_line()
)


# But the
# bottom row of the
# pairs plot
# further above
# shows that the values of `b` ($\beta$)
# around 0.5 have the highest `lp__` (log-probability)
# and that values of `a` ($\alpha$)
# around 1.0 have the highest `lp__`;
# 0.5 and 1.0 are indeed their true values.
# We often learn more when things go "wrong"
# and hopefully reading these diagnostic plots
# has helped you learn a little more about troubleshooting
# models that do not perfectly fit the noisy data!
# 
# ## ODE with a forcing function
# 
# ### Modified equations in Python
# 
# Finally!
# We get to discuss the main topic about
# how to use a forcing function
# when the equation of a state variable is unknown.
# Let's say that we did not know
# the equation terms for the predator state variable, `y`.
# Then, instead of two ODEs, we need to solve a single ODE.
# But we must replace the `y` term in the `x` equation
# with a fixed trajectory of `y`.
# 
# Before we had:
# 
# $$
# \begin{aligned}
# \frac{dx}{dt} &= \phantom{-} \alpha x - \beta x y \\
# \frac{dy}{dt} &= - \gamma y + \delta x y \\
# \end{aligned}
# $$
# 
# Now we have only the first equation and
# pretend we do not yet know the process that governs the second equation:
# 
# $$
# \frac{dx}{dt} = \phantom{-} \alpha x - \beta x y \\
# $$
# 
# We cannot fix `y` as a single value;
# `y` must vary in some way because `x` varies
# and `y` and `x` influence eachother.
# So we must supply a fixed trajectory of `y`
# based on the noisy data
# that only depends on time
# because `y` is no longer part of the system of equations
# that the ODE solver has access to.
# We must define an external function
# that produces a fixed trajectory of `y`.
# 
# Ideally,
# we would use a spline function.
# Splines are wonderful mathematical funtions
# because they're differentiable
# while still allowing us to fit complex, squiggly looking curves.
# However,
# spline functions are
# numerically harder to implement
# and Stan does not yet provide the mathematical primitives
# to easily add them.
# Therefore,
# we will use a simpler polynomial function:
# 
# $$
# \begin{aligned}
# \frac{dy}{dt} &= c_n t^n + c_{n-1} t^{n-1} \ldots + c_2 t^2 + c_1 t^1 + c_0 t^0 \\
# \implies
# \frac{dy}{dt} &= \sum_{i=0}^n c_i t^i \\
# \end{aligned}
# $$
# 
# Where,
# the polynomial terms are $t_0, t_1, t_2, \ldots, t_{n-1}, t_n$
# and their corresponding coefficients are $c_0, c_1, c_2 \ldots, c_{n-1}, c_n$.
# 
# > **Note:**
# > Unlike a spline function,
# > a polynomial function has trouble fitting periodic data.
# > See the plot below.

# In[20]:


def poly(x, degree):
    # Workaround statsmodels / patsy not supportting poly(variable, degree)
    # like R does.  See https://stackoverflow.com/a/77567246
    return np.vander(x, degree + 1, increasing=True)[:, 1:]

max_time = 7
max_degree = len(times[times <= max_time]) - 2
fits_poly = []
for i in range(1, max_degree + 1):
    model = smf.ols(f"y ~ poly(time, {i})",
                    data=ode_to_df(noisy).loc[times <= max_time, :])
    fits_poly += [model.fit()]


# In[21]:


data = (
    ode_to_df(noisy)
    .loc[:, ["time"]]
    .query(f"time <= {max_time}")
    .assign(**{
        re.findall(r"poly[(][^,]+, (\d+)[)]$",
                   fits_poly[i].model.formula)[0]:
        #f"{fits_poly[i].degree}":
        # We have force evaluate i here by running a (lambda i: [...])(i)
        # returning a lambda, otherwise all the predictions will use the last
        # value of i of the loop, because evaluation of i in the returned
        # lambda is deferred until .assign() is run.
        (lambda i: lambda df: (
            fits_poly[i]
            #.fit_transform(
            .predict(
                df)))(i)
        for i in range(len(fits_poly))
    })
)
data = pd.melt(
    data,
    id_vars=["time"],
    var_name="degree",
    value_name="y",
)
data["degree"] = pd.to_numeric(data["degree"])
print(data)
(
    ggplot(data, aes("time", "y")) +
    facet_wrap("degree") +
    geom_line() +
    geom_point(data=ode_to_df(noisy).query(f"time <= {max_time}"))
)


# In[22]:


fits_poly[-1].summary()


# In[23]:


fits_poly[0].pvalues.iloc[1]


# In[24]:


## Choose our polynomial degree.
criteria = pd.DataFrame(
    {
        "degree": [int(re.findall(r"poly[(][^,]+, (\d+)[)]$",
                                  fit.model.formula)[0])
                   for fit in fits_poly],
        "aic": [fit.aic for fit in fits_poly],
        "bic": [fit.bic for fit in fits_poly],
    })
print(criteria)
degree = 3
assert degree == criteria.query("degree > 1").reset_index().query("aic.argmax()").degree
assert degree == criteria.query("degree > 1").reset_index().query("bic.argmax()").degree
(
    ggplot(criteria.melt(id_vars=["degree"]),
           aes("degree", "value", group="variable", color="variable")) +
    geom_line() +
    geom_vline(xintercept=degree, linetype="longdash")
)


# In[25]:


# Now generate the ODE data using the forcing function (using t) and
# ignore any input for predator (dy_dt).
def y(t):
    """Return a positive scalar value for the predator y at time t.

    Importantly, this function is differentiable to help the ODE solver
    converge on a result.  Well, almost; there are discontinuities to keep
    the output positive, but the solver seems to tolerates these.

    Parameters
    ----------
    t : float
        Single timepoint at which to solve the differentiable equation.

    """
    return np.maximum(
        fits_poly[degree]
        .predict(pd.DataFrame({"time": t}, index=[0]))
        .values,
        np.array([0]),
    )


def lv_poly(t, y_, p=true_params):
    """Right-hand side of a Lotka–Volterra ODE with predator forcing function.

    Parameters
    ----------
    t : float
        Single timepoint at which to solve the system of equations.
    y_ : array_like
        State variables at the timepoint `t`.
    p : dict of float, optional
        Model parameters.

    """
    # Unpack y into convenience variances that must match the return value of
    # this function.
    x = y_[0]

    # Decode the initial parameters and initial state variable values.
    p = Namespace(**p)

    # Encode the equation terms.
    prey_growth = p.a * x
    prey_death = p.b * x * y(t)

    # Return the equation RHS.
    dx_dt = prey_growth - prey_death
    return (dx_dt, )


# In[26]:


fit_ode_ff = solve_ivp(fun=lv_poly,
                       t_span=[np.min(times), max_time],
                       # Preserve dimensions when subsetting to x by passing
                       # [1] as a list.
                       y0=list(noisy[1, [1]]),
                       t_eval=times[times <= max_time])


# In[27]:


data = pd.concat(
    (
        ode_to_df(noisy)
        .query(f"time <= {max_time}")
        .rename(columns=lambda col: col + "_orig" if col != "time" else col),
        ode_to_df(fit_ode_ff.y, times[times <= max_time]
        .reshape(-1, 1))
        .filter(["x"])
        .rename(columns={"x": "x_ff"}),
        pd.DataFrame({"y_ff": map(lambda t: y(t)[0], times[times <= max_time])})
    ),
    axis=1,
)
data = pd.melt(data, id_vars=["time"], var_name="variable_type")
data = pd.concat(
    (
        data.drop(columns=["variable_type"]),
        data.loc[:, "variable_type"]
        .str.split("_", expand=True)
        .rename(columns={0: "variable", 1: "type"}),
    ),
    axis=1)
data["type"] = pd.Categorical(data["type"],
                              categories=["orig", "ff"],
                              ordered=True)
(
    ggplot(data, aes("time", "value", group="variable", color="variable")) +
    facet_wrap("type", order=0) +
    geom_line() +
    geom_point(data=(
        data
        .query("type == 'orig'")
        .assign(type="ff")
    ))
)


# ### Fit the ODE parameters using the equivalent modified Stan model
# 
# We implement these equations in the Stan code.
# 
# The model we are going to fit
# is saved in a separate `lv-forcing-function.stan` file
# that looks like this:

# In[28]:


get_ipython().run_line_magic('stanf', 'lv-forcing-function.stan model_lv_ff')


# In[29]:


# Run Stan using the same forcing function.
fit_stan = model_lv_ff.sample(
    chains=4,
    parallel_chains=4,
    threads_per_chain=1,
    iter_warmup=1_000,
    iter_sampling=5_000,
    seed=12345,
    show_progress=False,
    data={
        # Must remove the row containing timepoint 0 and instead
        # provide it with the initial conditions.
        #
        # Dimensions:
        "T": len(times[(times > 0) & (times <= max_time)]),
        "V": noisy.T.shape[1] - 1,
        # Experimental data:
        "ts": times[(times > 0) & (times <= max_time)],
        "y": noisy.T[(times > 0) & (times <= max_time), 0].reshape(-1, 1),
        # Coefficients of fitted polynomial:
        "N_coef": len(fits_poly[degree].params.values),
        "coef": fits_poly[degree].params.values,
        # Initial conditions:
        "y0": noisy.T[0, [0]],
        "t0": 0,
    },
)


# ### Inspect the Stan forcing function fit

# In[30]:


fit_stan.summary()


# In[31]:


## Prepare data for plotting.
idata_stan = az.from_cmdstanpy(fit_stan)


# In[32]:


## Parameter plot of the fits.
azp.plot_forest(idata_stan)


# In[33]:


## Show all samples after warmup.
azp.plot_trace(idata1)


# In[34]:


## Show the pairs plot.
## FIXME: Possible to include lp__?
azp.plot_pair(idata_stan,
              visuals={
                  "contour": {"color": "black", "alpha": 0.3},
                  "scatter": {"size": 2, "alpha": 0.1},
             })


# In[35]:


## Plot the ODEs using a sample of 100 parameters from each of the chains.
df_fit_stan = fit_stan.draws_pd()
## Sample greater variation by only choosing one sample from each qcut n-tile.
qcut_20 = lambda x: pd.qcut(x, np.linspace(0, 1, 20), labels=False)
grouped = (
    df_fit_stan
    .filter(["chain__", "iter__", "a", "b"])
    .groupby("chain__")
)
n = 100
prng = np.random.Generator(np.random.MT19937(seed=123))
df_fit_stan_subset = (
    df_fit_stan.join(
        pd.DataFrame({
            "qa": grouped["a"].transform(qcut_20),
            "qb": grouped["b"].transform(qcut_20),
        })
    )
    .groupby(["chain__", "qa", "qb"])
    .sample(n=1, random_state=prng)
    ## Now sample the 100 parameters from each chain.
    .groupby("chain__")
    .sample(n=n, replace=True)
    .reset_index()
    .reset_index(names="sample")
)
n_ode = df_fit_stan_subset.shape[0]
arr = np.empty(np.array(pristine.y.T[times <= max_time].shape) * np.array([n_ode, 1]),
               dtype=pristine.y.dtype)
arr_y = np.array([list(map(lambda t: y(t)[0], times[times <= max_time]))]).T
for i in range(n_ode):
    ode_x = solve_ivp(fun=lv_poly,
                      t_span=[np.min(times), max_time],
                      y0=noisy.T[0, [0]],
                      t_eval=times[times <= max_time],
                      args=[
        {
            "a": df_fit_stan_subset.loc[i, "a"],
            "b": df_fit_stan_subset.loc[i, "b"],
        }
    ])
    assert ode_x.success
    begin = i * len(times[times <= max_time])
    end = (i + 1) * len(times[times <= max_time])
    arr[begin:end, [0]] = ode_x.y.T
    arr[begin:end, [1]] = arr_y

data = (
    pd.concat(
        (
            pd.DataFrame(
                {
                    "sample": np.repeat(range(n_ode), len(times[times <= max_time])),
                    "time": np.tile(times[times <= max_time], n_ode),
                }
            ),
            pd.DataFrame(arr, columns=true_y0.keys())
        ),
        axis=1
    )
    .join(df_fit_stan_subset.filter(["sample", "chain__"]),
          on="sample",
          lsuffix="l",
          rsuffix="r")
)
data = pd.melt(data, id_vars=["sample", "time", "chain__", "samplel", "sampler"])
data


# In[36]:


(
    ggplot(
        data, 
        aes(
            "time", 
            "value", 
            group=["sample", "variable"], 
            color="variable",
        )
    ) +
    facet_wrap("chain__") +
    geom_line(alpha=0.1) +
    geom_point(
        data=(
            pd.melt(
                ode_to_df(noisy)
                .query(f"time <= {max_time}"),
                id_vars=["time"],
            ).assign(sample=0)
            # The sample=0 is never used anywhere, but the plot will error out
            # without it.
        ),
        mapping=aes(
            group=[],
        ),
    )
)


# ## Finishing thoughts
# 
# 1. Don't simply choose a forcing function only based
#    on how it fits its own data;
#    instead,
#    choose a forcing function
#    by how well the other equations fit their data
#    by running the full ODE system.
# 
# 1. Spline functions are great as forcing functions,
#    but not always practical to use.
#    Polynomials can work,
#    but may not work well with some types of periodic data;
#    subset the data to timepoints
#    where the fit looks good.
# 
# 1. You may have noticed `seed=...` used in a few places.
#    This is to make the stochastic code outputs reproducible,
#    but one would typically not use these;
#    it is more common to use store the value of `np.random.get_state()`
#    to record a seed used for a simulation
#    than to force a specific outcome
#    because well-adjusted simulations should produce
#    very similar results
#    in spite of pseudo-random number generator initial states.
