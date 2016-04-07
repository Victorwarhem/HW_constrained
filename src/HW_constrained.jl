

# constrained maximization exercises

## portfolio choice problem

module HW_constrained

	using JuMP, NLopt, DataFrames

	export data, table_NLopt, table_JuMP, runAll, runAll2

	#before all, let's define the utility function and its derivate wrt a:
	u(ab,c)=-exp(-ab*c)
	duda(ab,c)=ab*exp(-ab*c)

	function data(a=0.5)
	n=3
	p=[1.0;1.;1.]
	e=[2.0;0;0]
	z1=[1.0;1.0;1.0;1.0]
	z2=[0.72;0.92;1.12;1.32]
	z3=[0.86;0.96;1.06;1.16]
	z=hcat(repeat(z1,inner=[4],outer=[1]),repeat(z2,inner=[4],outer=[1]),repeat(z3,inner=[1],outer=[4]))
	pis=Float64[1/16 for i=1:16]
	return Dict("a"=>a,"n"=>n,"p"=>p,"e"=>e,"z"=>z,"pi"=>pis)
	end

	function max_JuMP(a=0.5)
	d=data(a)
	mod=Model()
	@defVar(mod,c>=0.0)
	@defVar(mod,omega[1:3])
	@setNLObjective(mod, Max,-exp(-a*c)+sum{d["pi"][j]*(-exp(-a*sum{omega[i]*d["z"][j,i],i=1:3})),j=1:16})
	@addNLConstraint(mod,c+sum{d["p"][i]*(omega[i]-d["e"][i]),i=1:d["n"]}==0.0)
	print(mod)
	solve(mod)
	return Dict("val"=>getObjectiveValue(mod),"c"=>getValue(c),"omega"=>getValue(omega))
	end

	function table_JuMP()
	df=DataFrame(a=[0.5;1.0;5.0],c=[0.;0.;0.],omega1=[0.;0.;0.],omega2=[0.;0.;0.],omega3=[0.;0.;0.],func=[0.;0.;0.])
	for i in 1:nrow(df)
	results=max_JuMP(df[i,:a])
	df[i,:c]=results["c"]
	df[i,:omega1]=results["omega"][1]
	df[i,:omega2]=results["omega"][2]
	df[i,:omega3]=results["omega"][3]
	df[i,:func]=results["val"]
	end
	return df
	end
	d=data()

	function obj(results::Vector,grad::Vector,d::Dict)
	c=results[1]
	omega=results[2:4]
	for i in 1:16
		next=1/16*u(d["a"],dot(omega,vec(d["z"][i,:])))
	end

	if length(grad)>0.
	grad[1]=duda(d["a"],c)
	for i in 1:3
		for j in 1:16
		grad[i+1]+=1/16*duda(d["a"],dot(omega,vec(d["z"][j,:])))*d["z"][j,i]
		end
	end
	return utils=u(d["a"],c)+next
	println("value of utility: $utils")
	end

	function constr(x::Vector,grad::Vector,d::Dict)
	constraint=d["p"]*(d["e"]-x[2:4])-x[1]
	grad[1]=-1
	grad[2:4]=-d["p"]
	return constraint
	end

	function max_NLopt(a=0.5)
	d=data()
	optim=NLopt.Opt(:LD_SLSQP,4)
	max_objective!(optim,(x,grad)->obj(x,grad,d))
	lower_bounds!(optim,[0;[-Inf for i in 1:3]])
	upper_bounds!(optim,[+Inf for i in 1:4])
	equality_constraint!(optim,(x,grad)->constr(x,grad,d),1e-7)
	ftol_rel!(optim,1e-9)
	(optimfunc,optimx,ret)=optimize(optim,rand(4))
	end

	function table_NLopt()
	df=DataFrame(a=[0.5;1.0;5.0],c=[0.;0.;0.],omega1=[0.;0.;0.],omega2=[0.;0.;0.],omega3=[0.;0.;0.],func=[0.;0.;0.])
	for i in 1:d["n"]
	results=max_NLopt(df[i,:a])
		for j in 2:ncol(df)-1
			d[i,j]=results[2][j-1]
		end
		d[i,end]=results[1]
	end
	return df
	end

	# function `f` is for the NLopt interface, i.e.
	# it has 2 arguments `x` and `grad`, where `grad` is
	# modified in place
	# if you want to call `f` with more than those 2 args, you need to
	# specify an anonymous function as in
	# other_arg = 3.3
	# test_finite_diff((x,g)->f(x,g,other_arg), x )
	# this function cycles through all dimensions of `f` and applies
	# the finite differencing to each. it prints some nice output.
	function test_finite_diff(f::Function,x::Vector{Float64},tol=1e-6)
		g=similar(x)
		y=f(x,g)
		fd=finite_diff(f,x)
		table=hcat(1:length(x),g,fd,abs(g-fd))
		err=find(abs(g-fd).>tol)
		if length(err)>0
			return err
			return false
		else
		prinln("everything is okay")
			return true
		end
	end

	# do this for each dimension of x
	# low-level function doing the actual finite difference
	function finite_diff(f::Function,x::Vector)
	end

	function runAll2()
		table_JuMP()
		table_NLopt()
		test_finite_diff()
		finite_diff()
	end

	function runAll()
		println("running tests:")
		include("test/runtests.jl")
		println("")
		println("JumP:")
		table_JuMP()
		println("")
		println("NLopt:")
		table_NLopt()
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end
end
