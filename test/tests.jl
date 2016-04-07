


module AssetTests

	using FactCheck, HW_constrained

		facts("tests gradient of objective function") do
		d = HW_constrained.data(0.5)
 			x = ones(Float64,d["n"]+1)
 			@fact HW_constrained.test_finite_diff((ix,g)->HW_constrained.obj(ix,g,d),x) --> true
 
 			x = rand(d["n"]+1)
 			@fact HW_constrained.test_finite_diff((ix,g)->HW_constrained.obj(ix,g,d),x) --> true	
		end


		facts("tests gradient of constraint function") do
d = HW_constrained.data(0.5)
 			x = ones(Float64,d["n"]+1)
 			@fact HW_constrained.test_finite_diff((ix,g)->HW_constrained.constr(ix,g,d),x) --> true
 
 			x = rand(d["n"]+1)
 			@fact HW_constrained.test_finite_diff((ix,g)->HW_constrained.constr(ix,g,d),x) --> true
		end
	end

	context("testing result of both maximization methods") do
truth = HW_constrained.DataFrame(a=[0.5;1.0;5.0],c = [1.008;1.004;1.0008],omega1=[-1.41237;-0.20618;0.758763],omega2=[0.801455;0.400728;0.0801455],omega3=[1.60291;0.801455;0.160291],func=[-1.20821;-0.732819;-0.013422])
 
 		facts("checking result of NLopt maximization") do
 
 			t1 = table_NLopt()
 			for c in names(truth)
 				@fact t1[c] --> roughly(truth[c],atol=1.e-4)
 			end
 		end
 
 
 		facts("checking result of NLopt maximization") do
 			t1 = table_JuMP()
 			for c in names(truth)
 				@fact t1[c] --> roughly(truth[c],atol=1.e-4)
 			end
 		end
		
	end

end



