function Initial5Body(T=Float64)
GmSun = 0.295912208285591100e-3
GmMercury = 0.491248045036476000e-10
GmVenus = 0.724345233264412000e-9
GmEarth = 0.888769244512563400e-9
GmMoon = 0.109318945074237400e-10
GmMars = 0.954954869555077000e-10

Gm = [GmSun + GmMercury + GmVenus+ GmEarth + GmMoon + GmMars, 0.282534584083387000e-6,
  0.845970607324503000e-7, 0.129202482578296000e-7,
  0.152435734788511000e-7]


q = [0.00450250878464055477, 0.00076707642709100705,
   0.00026605791776697764,
   -5.37970676855393644523, -0.83048132656339789295,
-0.22482887442656542236,
   7.89439068290953155477, 4.59647805517127300705,
   1.55869584283189997764,
   -18.26540225387235944523, -1.16195541867586999295,
-0.25010605772133802236,
   -16.05503578023336944523, -23.94219155985470899295,
-9.40015796880239402236
]

v = [-0.00000035174953607552, 0.00000517762640983341,
   0.00000222910217891203,
       0.00109201259423733748, -0.00651811661280738459,
-0.00282078276229867897,
      -0.00321755651650091552, 0.00433581034174662541,
   0.00192864631686015503,
      0.00022119039101561468, -0.00376247500810884459,
-0.00165101502742994997,
      0.00264276984798005548, -0.00149831255054097759,
-0.00067904196080291327
]

qq = copy(q)
vv = copy(v)

q0 = reshape(qq,3,:)
v0 = reshape(vv,3,:)

q0bar = [sum(Gm .* q0[j,:])/sum(Gm) for j in 1:3]
v0bar = [sum(Gm .* v0[j,:])/sum(Gm) for j in 1:3]

q0 = q0 .- q0bar
v0 = v0 .- v0bar


N = length(Gm)    

u0 = Array{T}(undef,3,N,2)
u0[:,:,2] .= v0[:,1:N]
u0[:,:,1] .= q0[:,1:N]
bodylist = ["Sun" "Jupiter" "Saturn" "Uranus"  "Neptune"]

return u0, Gm, bodylist
end
