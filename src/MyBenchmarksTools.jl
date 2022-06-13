struct WPTests
   errors::Array{Float64,1}
   times::Array{Float64,1}
end


function launch_IRKGL_seq_tests(final_state, prob, s, dts; initial_interp=true, itermax=100, nruns=10)

#
#    @belapsed erabiltzea problemak ematen ditu!!!
#
     k=length(dts)
     errors=zeros(k)
     times=zeros(k)

     @unpack f,u0,tspan,p,kwargs=prob
     t0=tspan[1]
     tf=tspan[2]

     for i in 1:k

       dti=dts[i]
       n=Int64((tf-t0)/dti)
       # m=n => save_everystep=false
       soli=IRKGL_Seq(s, u0, t0, tf, 1, n, f, p, initial_interp=initial_interp, itermax=itermax)
       errors[i]=norm(final_state-soli.u[end])/norm(final_state)

       for k in 1:nruns
           times[i]+=@elapsed IRKGL_Seq(s, u0, t0, tf, 1, n, f, p, initial_interp=initial_interp, itermax=itermax)
       end

       times[i]=times[i]/nruns

     end

     WPTests(errors,times)

end


function launch_IRKGL_simd_tests(final_state, prob, dim, s, dts; initial_interp=-1, partitioned=false, floatType=Float64, maxiters=100, nruns=10)

#
#    @belapsed erabiltzea problemak ematen ditu!!!
#
     k=length(dts)
     errors=zeros(k)
     times=zeros(k)

     for i in 1:k

       dti=dts[i]
       soli=solve(prob,IRKGL_simd(s=s, partitioned=partitioned, initial_interp=initial_interp, dim=dim, floatType=floatType);
                  dt=dti, save_everystep=false, maxiters=maxiters)
       errors[i]=norm(final_state-soli.u[end])/norm(final_state)

       for k in 1:nruns
           times[i]+=@elapsed solve(prob,IRKGL_simd(s=s, partitioned=partitioned, initial_interp=initial_interp, dim=dim, floatType=floatType);
                      dt=dti, save_everystep=false, maxiters=maxiters)
       end

       times[i]=times[i]/nruns

     end

     WPTests(errors,times)

end

function launch_method_tests(method, final_state, prob, launch_list; adaptive=true, maxiters=10^9,  nruns=10)

#
#    @belapsed erabiltzea problemak ematen ditu!!!
#

     k=length(launch_list)
     errors=zeros(k)
     times=zeros(k)

     if (adaptive==true)

       tols=launch_list

       for i in 1:k

           tol=tols[i]
           soli= solve(prob, method, abstol=tol, reltol=tol, adaptive=true, save_everystep=false, dense=false, maxiters=maxiters);
           errors[i]=norm(final_state-soli.u[end])/norm(final_state)

           for k in 1:nruns
               times[i]+=@elapsed solve(prob, method, abstol=tol, reltol=tol, adaptive=true, save_everystep=false, dense=false, maxiters=maxiters);
           end

           times[i]=times[i]/nruns

     end

    else # adaptive_false

      dts=launch_list

      for i in 1:k

          dti=dts[i]
          soli= solve(prob, method, dt=dti, adaptive=false, save_everystep=false, dense=false)
          errors[i]=norm(final_state-soli.u[end])/norm(final_state)

          for k in 1:nruns
              times[i]+=@elapsed solve(prob, method, dt=dti, adaptive=false, save_everystep=false, dense=false);
          end

          times[i]=times[i]/nruns

      end

    end

    WPTests(errors,times)

end
