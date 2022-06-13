
include("IRKGL_Coefficients.jl")
include("VecArray_def.jl")

struct IRKGL_SIMD_Cache{floatType,fType,pType,s_,dim,dim_}
    odef::fType # function defining the ODE system
    p::pType # parameters and so
    b::Vec{s_,floatType}
    c::Vec{s_,floatType}
    mu::VecArray{s_,floatType,2}
    nu::VecArray{s_,floatType,2}
    nu1::VecArray{s_,floatType,2}
    nu2::VecArray{s_,floatType,2}
    U::VecArray{s_,floatType,dim_}
    Uz::VecArray{s_,floatType,dim_}
    L::VecArray{s_,floatType,dim_}
    Lz::VecArray{s_,floatType,dim_}
    F::VecArray{s_,floatType,dim_}
    Dmin::Array{floatType,dim}
    maxiters::Int64
    step_number::Array{Int64,0}
    first3iters::Array{Int64,1}
    initial_interp::Array{Int64,0}
    length_u::Int64
end



struct IRKNGL_SIMD_Cache{floatType,fType,pType,s_,dim,dim_}
    odef::fType # function defining the ODE system
    p::pType # parameters and so
    b::Vec{s_,floatType}
    c::Vec{s_,floatType}
    mu::VecArray{s_,floatType,2}
    nu::VecArray{s_,floatType,2}
    nu1::VecArray{s_,floatType,2}
    nu2::VecArray{s_,floatType,2}
    U::VecArray{s_,floatType,dim_}
    Uz::VecArray{s_,floatType,dim_}
    L::VecArray{s_,floatType,dim_}
    Lz::VecArray{s_,floatType,dim_}
    F::VecArray{s_,floatType,dim_}
    Dmin::Array{floatType,dim}
    maxiters::Int64
    step_number::Array{Int64,0}
    first3iters::Array{Int64,1}
    initial_interp::Array{Int64,0}
    length_u::Int64
    length_q::Int64
end


abstract type IRKAlgorithm{s,partitioned,initial_interp, dim,floatType,m,myoutputs} <: OrdinaryDiffEqAlgorithm end
struct IRKGL_simd{s,partitioned, initial_interp, dim,floatType,m,myoutputs} <: IRKAlgorithm{s, partitioned, initial_interp, dim,floatType,m,myoutputs} end
IRKGL_simd(;s=8,partitioned=false, initial_interp=-1, dim=1,floatType=Float64,m=1,myoutputs=false)=IRKGL_simd{s,partitioned, initial_interp, dim,floatType,m,myoutputs}()

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem{uType,tType,isinplace},
    alg::IRKGL_simd{s,partitioned,initial_interp, dim,floatType, m,myoutputs}, args...;
    dt,
    save_everystep=true,
    adaptive=false,
    maxiters=100,
    kwargs...) where {floatType<: Union{Float32,Float64},uType,tType,isinplace,dim,s,m,partitioned,initial_interp,myoutputs}


    destats = DiffEqBase.DEStats(0)
    @unpack f,u0,tspan,p,kwargs=prob
    t0=tspan[1]
    tf=tspan[2]
    tType2=eltype(tspan)
    utype = Vector{floatType}
    ttype = floatType

    step_number = Array{Int64,0}(undef)
    step_number[] = 0
    init_interp =  Array{Int64,0}(undef)
    init_interp[] = initial_interp
    first3iters=[0,0,0]

    dts = Array{tType2}(undef, 1)
    if (adaptive==false)
       dtprev=dt
    else
       dtprev=zero(tType2)
    end

    sdt = sign(dt)
    dts=[dt,dtprev,sdt]

    (b_, c_, mu_, nu_, nu1_, nu2_) = IRKGLCoefficients(s,dt)
    length_u = length(u0)
    dims = size(u0)

    c = vload(Vec{s,floatType}, c_, 1)
    b = vload(Vec{s,floatType}, b_, 1)
    nu=VecArray{s,floatType,2}(nu_)
    nu1=VecArray{s,floatType,2}(nu1_)
    nu2=VecArray{s,floatType,2}(nu2_)
    mu=VecArray{s,floatType,2}(mu_)

     ej=zero(u0)

     u0type=typeof(u0)
     uu = u0type[]
     tt = ttype[]

     zz=zeros(Float64, s, dims...)
     U=VecArray{s,Float64,length(dims)+1}(zz)
     L=deepcopy(U)
     Lz=deepcopy(U)
     F=deepcopy(U)
     Uz=deepcopy(U)

    Dmin = zero(u0)

   if partitioned
      length_q = div(length_u,2)
      irknglcache = IRKNGL_SIMD_Cache(f,p,b,c,mu,nu,nu1,nu2,U,Uz,L,Lz,
                               F,Dmin,maxiters, step_number,first3iters,
                               init_interp,length_u,length_q)
   else
     irkglcache = IRKGL_SIMD_Cache(f,p,b,c,mu,nu,nu1,nu2,U,Uz,L,Lz,
                               F,Dmin,maxiters,step_number,first3iters,
                               init_interp,length_u)
   end

  iters = Float64[]

  push!(uu,u0)
  push!(tt,t0)
  push!(iters,0.)

  tj = [t0, zero(t0)]
  uj = copy(u0)

  cont=true
  j=0


  if partitioned

    while cont
      tit=0
      k=0

      for i in 1:m
         j+=1
         k+=1

         irknglcache.step_number[] += 1
         j_iter = IRKNGLstep!(tj,uj,ej,prob,dts,irknglcache)

         if (irknglcache.initial_interp[]==-1)
             if (irknglcache.step_number[]<3)
                 irknglcache.first3iters[j]=j_iter
             elseif (irknglcache.step_number[]==3)
                 irknglcache.first3iters[j]=j_iter
                 irknglcache.initial_interp[]=argmin(irknglcache.first3iters)-1
             end
        end


         tit+= j_iter
         if (dts[1]==0)
             cont=false
             break
         end
      end

      if save_everystep !=false || (cont==false)
          push!(uu,uj+ej)
          push!(tt,tj[1]+tj[2])
          push!(iters, tit/k)
      end
    end

  else

    while cont

      tit=0
      k=0

      for i in 1:m
         j+=1
         k+=1

         irkglcache.step_number[] += 1
         j_iter = IRKGLstep!(tj,uj,ej,prob, dts,irkglcache)

         if (irkglcache.initial_interp[]==-1)
             if (irkglcache.step_number[]<3)
                 irkglcache.first3iters[j]=j_iter
             elseif (irkglcache.step_number[]==3)
                 irkglcache.first3iters[j]=j_iter
                 irkglcache.initial_interp[]=argmin(irkglcache.first3iters)-1
             end
        end

         tit+= j_iter
         if (dts[1]==0)
            cont=false
            break
        end
      end

      if save_everystep !=false || (cont==false)
          push!(uu,uj+ej)
          push!(tt,tj[1]+tj[2])
          push!(iters, tit/k)
      end
    end
  end


  sol=DiffEqBase.build_solution(prob,alg,tt,uu,destats=destats,retcode= :Success)

  if (myoutputs==true)
      return(sol,iters)
  else
      return(sol)
  end

  return(sol)

  end


  function IRKGLstep!(ttj,uj,ej,prob,dts,irkglcache::IRKGL_SIMD_Cache{floatType,fType,pType,s_,dim,dim_}) where {floatType,fType,pType,s_,dim,dim_}

     f = irkglcache.odef
     p = irkglcache.p
     b = irkglcache.b
     c = irkglcache.c
     mu = irkglcache.mu
     nu = irkglcache.nu
     nu1 = irkglcache.nu1
     nu2 = irkglcache.nu2
     U = irkglcache.U
     Uz = irkglcache.Uz
     L = irkglcache.L
     Lz = irkglcache.Lz
     F = irkglcache.F
     Dmin = irkglcache.Dmin
     step_number = irkglcache.step_number[]
     first3iters = irkglcache.first3iters
     initial_interp = irkglcache.initial_interp[]
     len = irkglcache.length_u
     s = length(b)
     maxiters = (step_number==1 ? 10+irkglcache.maxiters : irkglcache.maxiters )
     tj = ttj[1]
     te = ttj[2]
     indices=1:len
     flzero = zero(floatType)

     dt=dts[1]
     dtprev=dts[2]
     sdt=dts[3]
     tf=prob.tspan[2]


     if (initial_interp==1 || (initial_interp==2 && step_number[]==2) || (initial_interp==-1 && step_number[]==2))
       for k in indices
          Lk = getindex_(L,k)
          dUk = muladd(nu[1], Lk[1], ej[k])
          for is in 2:s
              dUk = muladd(nu[is], Lk[is], dUk)
          end
          setindex_!(U, uj[k]+dUk, k)
       end

    elseif ( (initial_interp==2 && step_number[]>2) || (initial_interp==-1 && step_number[]==3))
        for k in indices
           Lk = getindex_(L,k)
           dUk = muladd(nu1[1], Lk[1], ej[k])
           for is in 2:s
               dUk = muladd(nu1[is], Lk[is], dUk)
           end
           Lkz = getindex_(Lz,k)
           for is in 1:s
               dUk = muladd(nu2[is], Lkz[is], dUk)
           end
           setindex_!(U, uj[k]+dUk, k)
        end
     else
        for k in indices
           uej = uj[k] + ej[k]
          setindex_!(U, uej, k)
        end
     end


  j_iter = 0  # counter of fixed_point iterations
  Dmin .= Inf

  iter = true # Initialize iter outside the for loop
  plusIt=true
  diffU = true

  Lz.data .= L.data

  @inbounds while (j_iter<maxiters && iter)

          iter = false

          Uz.data .= U.data
          j_iter += 1

          f(F, U, p, tj + dt*c)

          diffU = false

          for k in indices
              Fk = getindex_(F,k)
              Lk = dt*(b*Fk)
              dUk = muladd(mu[1], Lk[1], ej[k])
              for is in 2:s
                      dUk = muladd(mu[is], Lk[is], dUk)
              end
              Uk = uj[k]+dUk
              setindex_!(U, Uk, k)
              setindex_!(L, Lk, k)
              Uzk = getindex_(Uz,k)
              DY = maximum(abs(Uk-Uzk))

              if DY>0
                  diffU = true
                  if DY< Dmin[k]
                     Dmin[k]=DY
                     iter=true
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


      @inbounds if (j_iter<maxiters && diffU)
              j_iter += 1
              f(F, U, p, tj + dt*c)

              for k in indices
                  Fk = getindex_(F, k)
                  Lk = dt*(b*Fk)
                  setindex_!(L, Lk, k)
              end
      end


      @inbounds for k in indices    #Batura konpentsatuaren parekoa
          Lk = getindex_(L,k)
          L_sum = sum(Lk)
          res = Base.TwicePrecision(uj[k], ej[k]) + L_sum
          uj[k] = res.hi
          ej[k] = res.lo
       end


       res = Base.TwicePrecision(tj, te) + dt
       ttj[1] = res.hi
       ttj[2] = res.lo

       dts[1]=sdt*min(abs(dt),abs(tf-(ttj[1]+ttj[2])))
       dts[2]=dt

       return  (j_iter)


  end



  function IRKNGLstep!(ttj,uj,ej,prob,dts, irknglcache::IRKNGL_SIMD_Cache{floatType,fType,pType,s_,dim,dim_}) where {floatType,fType,pType,s_,dim,dim_}

     f = irknglcache.odef
     p = irknglcache.p
     b = irknglcache.b
     c = irknglcache.c
     mu = irknglcache.mu
     nu = irknglcache.nu
     nu1 = irknglcache.nu1
     nu2 = irknglcache.nu2
     U = irknglcache.U
     Uz = irknglcache.Uz
     L = irknglcache.L
     Lz = irknglcache.Lz
     F = irknglcache.F
     Dmin = irknglcache.Dmin
     step_number = irknglcache.step_number[]
     first3iters = irknglcache.first3iters
     initial_interp = irknglcache.initial_interp[]
     len = irknglcache.length_u
     lenq = irknglcache.length_q
     s = length(b)
     len = length(uj)
     maxiters = (step_number==1 ? 10+irknglcache.maxiters : irknglcache.maxiters )
     tj = ttj[1]
     te = ttj[2]
     flzero = zero(floatType)

     dt=dts[1]
     dtprev=dts[2]
     sdt=dts[3]
     tf=prob.tspan[2]

     indices=1:len
     indices1 = 1:lenq
     indices2 = (lenq+1):len


   if (initial_interp==1 || (initial_interp==2 && step_number[]==2) || (initial_interp==-1 && step_number[]==2))
       for k in indices
          Lk = getindex_(L,k)
          dUk = muladd(nu[1], Lk[1], ej[k])
          for is in 2:s
              dUk = muladd(nu[is], Lk[is], dUk)
          end
          setindex_!(U, uj[k]+dUk, k)
       end

   elseif ( (initial_interp==2 && step_number[]>2) || (initial_interp==-1 && step_number[]==3))
       for k in indices
          Lk = getindex_(L,k)
          dUk = muladd(nu1[1], Lk[1], ej[k])
          for is in 2:s
              dUk = muladd(nu1[is], Lk[is], dUk)
          end
          Lkz = getindex_(Lz,k)
          for is in 1:s
              dUk = muladd(nu2[is], Lkz[is], dUk)
          end
          setindex_!(U, uj[k]+dUk, k)
       end

     else
        for k in indices
          uej = uj[k] + ej[k]
          setindex_!(U, uej, k)
        end
     end


      j_iter = 0  # counter of fixed_point iterations

      Dmin .= Inf

  iter = true # Initialize iter outside the for loop
  plusIt=true
  diffU = true


  Lz.data .= L.data

  @inbounds while (j_iter<maxiters && iter)

          iter = false

          Uz.data .= U.data

          j_iter += 1

          f(F, U, p, tj + dt*c, 1)

          for k in indices1
                  Fk = getindex_(F,k)
                  Lk =dt*(b*Fk)
                  setindex_!(L, Lk, k)
                  dUk = muladd(mu[1], Lk[1], ej[k])
                  for is in 2:s
                      dUk = muladd(mu[is], Lk[is], dUk)
                  end
                  setindex_!(U, uj[k] + dUk, k)
          end

          f(F, U, p, tj + dt*c, 2)

          for k in indices2
                  Fk = getindex_(F,k)
                  Lk = dt*(b*Fk)
                  setindex_!(L, Lk, k)
                  dUk = muladd(mu[1], Lk[1], ej[k])
                  for is in 2:s
                      dUk = muladd(mu[is], Lk[is], dUk)
                  end
                  setindex_!(U, uj[k]+dUk, k)
          end


         diffU = false

         for k in indices   # Hemen indices1 jarri liteke, q'=v, v'=f(q,t) moduko ED-a dela suposatuz

              Uk = getindex_(U,k)
              Uzk = getindex_(Uz,k)
              DY = maximum(abs(Uk-Uzk))

              if DY>0
                  diffU = true
                  if DY< Dmin[k]
                     Dmin[k]=DY
                     iter=true
                  end
              end
          end

          if (!iter && diffU && plusIt)  #
              iter=true
              plusIt=false
          else
              plusIt=true
          end

      end # while


      @inbounds if (j_iter<maxiters && diffU)

          j_iter += 1

          f(F, U, p, tj + dt*c, 1)

          for k in indices1
                  Fk = getindex_(F,k)
                  Lk = dt*(b*Fk)
                  dUk = muladd(mu[1], Lk[1], ej[k])
                  for is in 2:s
                      dUk = muladd(mu[is], Lk[is], dUk)
                  end
                  setindex_!(U, uj[k]+dUk, k)
                  setindex_!(L, Lk, k)
          end

          f(F, U, p, tj + dt*c, 2)

          for k in indices2
                  Fk = getindex_(F,k)
                  Lk = dt*(b*Fk)
                  setindex_!(L, Lk, k)
          end

      end


      @inbounds for k in indices    #Batura konpentsatuaren parekoa
          Lk = getindex_(L,k)
          L_sum = sum(Lk)
          res = Base.TwicePrecision(uj[k], ej[k]) + L_sum
          uj[k] = res.hi
          ej[k] = res.lo
       end

       res = Base.TwicePrecision(tj, te) + dt
       ttj[1] = res.hi
       ttj[2] = res.lo

       dts[1]=sdt*min(abs(dt),abs(tf-(ttj[1]+ttj[2])))
       dts[2]=dt


       return  (j_iter)

  end
