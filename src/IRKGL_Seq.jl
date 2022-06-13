include("IRKGL_Coefficients.jl")

struct NireODESol{uType,tType,fType}
   t::Array{tType,1}
   u::Array{uType,1}
   iter::Array{Float64,1}
   retcode::Symbol
   f::fType
end


struct IRKGL_Seq_Cache{uType,tType,fType,pType}
    odef::fType # function defining the ODE system
    p::pType # parameters and so
    h::tType
    b::Vector{tType}
    c::Vector{tType}
    mu::Array{tType,2}
    nu::Array{tType,2}
    U::Vector{uType}
    Uz::Vector{uType}
    L::Vector{uType}
    F::Vector{uType}
    Dmin::Vector{uType}
    itermax::Int64
    step_number::Array{Int64,0}
    initial_interp::Bool
    trace::Bool
end

function  IRKGL_Seq(s::Int64, u0::utype, t0::ttype, tf::ttype, n, m, f::ftype, p::ptype;  initial_interp=true,  itermax=100, trace=false) where {utype, ttype, ftype, ptype}
    step_number = Array{Int64,0}(undef)
    step_number[] = 0
    h = (tf-t0)/(n*m)

    (b, c, mu, nu) = IRKGLCoefficients(s,h)
    sm = s*m
    U = Array{utype}(undef, s)
    Uz = Array{utype}(undef, s)
    L = Array{utype}(undef, s)
    Dmin=Array{utype}(undef,s)
    F = Array{utype}(undef, s)
    for i in 1:s
        U[i] = zero(u0)
        Uz[i] = zero(u0)
        L[i] = zero(u0)
        Dmin[i] = zero(u0)
        F[i] = zero(u0)
    end

    ej=zero(u0)


    uu = Array{typeof(u0)}(undef, n+1)
    tt = Array{ttype}(undef, n+1)
    irkglcache = IRKGL_Seq_Cache(f,p,h,b,c,mu,nu,U,Uz,L,F,Dmin,itermax,step_number,initial_interp,trace)
    iters = zeros(n+1)
    uu[1] = copy(u0)
    tt[1] = t0
    tj = [t0, zero(t0)]
    uj = copy(u0)
    cont = true
    for j in 1:n
        for i in 1:m
           irkglcache.step_number[] += 1
           j_eval = IRKGLstep!(tj,uj,ej,irkglcache)
           iters[j+1] += j_eval
        end
        iters[j+1] /= sm  # average number of iterations per step
        uu[j+1] = uj + ej
        tt[j+1] = tj[1] + tj[2]
    end
    sol = NireODESol(tt,uu,iters,:Successs,f)
    return sol
  end




function IRKGLstep!(ttj,uj,ej,IRKGL_Seq_Cache::IRKGL_Seq_Cache)
       f = IRKGL_Seq_Cache.odef
       p = IRKGL_Seq_Cache.p
       h = IRKGL_Seq_Cache.h
       b = IRKGL_Seq_Cache.b
       c = IRKGL_Seq_Cache.c
       mu = IRKGL_Seq_Cache.mu
       nu = IRKGL_Seq_Cache.nu
       U = IRKGL_Seq_Cache.U
       Uz = IRKGL_Seq_Cache.Uz
       L = IRKGL_Seq_Cache.L
       F = IRKGL_Seq_Cache.F
       Dmin = IRKGL_Seq_Cache.Dmin
       step_number = IRKGL_Seq_Cache.step_number[]
       initial_interp = IRKGL_Seq_Cache.initial_interp
       trace = IRKGL_Seq_Cache.trace
       s = length(b)
       dim = length(uj)
       elems = s*dim
       itermax = (step_number==1 ? 10+IRKGL_Seq_Cache.itermax : IRKGL_Seq_Cache.itermax )
       sitermax = s * itermax
       tj = ttj[1]
       te = ttj[2]
       indices=eachindex(uj)

       if initial_interp
          for is in 1:s
            for k in indices
                dUik = muladd(nu[is,1], L[1][k], ej[k])
                for js in 2:s
                    dUik = muladd(nu[is,js], L[js][k], dUik)
                end
                U[is][k] =  uj[k]  + dUik
            end
          end
       else
          for is in 1:s
             for k in indices
                U[is][k] = uj[k] + ej[k]
             end
          end
       end


    j_eval = 0  # counter of evaluations of f
    j_iter = 0  # counter of fixed_point iterations


    @inbounds for is in 1:s
            f(F[is], U[is], p, tj + h*c[is])
            for k in indices
                L[is][k] = h*b[is]*F[is][k]
                Dmin[is][k] = Inf
            end

    end
    j_eval += s


    iter = true # Initialize iter outside the for loop
    plusIt=true



    @inbounds while (j_eval<sitermax && iter)
            j_iter += 1

            iter = false

            for is in 1:s
              for k in indices
                Uz[is][k] = U[is][k]
                dUik = muladd(mu[is,1], L[1][k], ej[k])
                for js in 2:s
                    dUik = muladd(mu[is,js], L[js][k], dUik)
                end
                U[is][k] =  uj[k] + dUik
              end
            end


            if trace
                DYmax = 0.
                ismax = 0
                kmax = 0
                println("step_number=$step_number, j_iter=$j_iter,  initial_fp=false")
            end
            diffU = false
            for is in 1:s

                eval=false
                for k in indices
                            DY = abs(U[is][k]-Uz[is][k])

                            if DY>0.
                               eval = true
                               diffU = true
                               if DY< Dmin[is][k]
                                  Dmin[is][k]=DY
                                  iter=true
                               end
                           end
                end


               if eval

                    f(F[is], U[is], p,  tj  + h*c[is])

                    j_eval += 1
                    for k in indices
                        L[is][k] = h*b[is]*F[is][k]
                    end
               end
           end


            if (!iter && diffU && plusIt)
                iter=true
                plusIt=false
            else
                plusIt=true
            end

        end # while



        for k in indices    #Batura konpentsatuaren parekoa
            e0 = ej[k]
            for is in 1:s
	         e0 += muladd(F[is][k], h*b[is], -L[is][k])
            end
            res = Base.TwicePrecision(uj[k], e0)
            for is in 1:s
	       res += L[is][k]
            end
            uj[k] = res.hi
            ej[k] = res.lo
         end


         res = Base.TwicePrecision(tj, te) + h
         ttj[1] = res.hi
         ttj[2] = res.lo
         if trace
            println("j_eval=", j_eval)
         end
         return  (j_eval)

end


function Rdigits(x::Real,r::Real)
    mx=r*x
    mxx=mx+x
    return mxx-mx
end
