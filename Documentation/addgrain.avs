setmemorymax(1024)
#loadplugin(".\AddGrainC.dll")
#loadplugin(".\Debug\AddGrainC.dll")
#loadplugin(".\Release\AddGrainC.dll")
#loadplugin(".\RelSym\AddGrainC.dll")

w = 380
h = 180

yv12 = blankclip(pixel_type="yv12" ,color=$808080,width=w,height=h,length=10).loop(1000000)
yuy2 = blankclip(pixel_type="yuy2" ,color=$808080,width=w,height=h,length=10).loop(1000000)
rgb =  blankclip(pixel_type="rgb24",color=$808080,width=w,height=h,length=10).loop(1000000)
rgba = blankclip(pixel_type="rgb32",color=$808080,width=w,height=h,length=10).loop(1000000)

stackhorizontal ( \
	stackvertical(yv12.addgrainc(200,200).ConvertToRGB32(), \
		yuy2.addgrainc(200,200,seed=12).ConvertToRGB32(), \
		yuy2.addgrainc(200,200).ConvertToRGB32(), \
		rgb.addgrainc(200,200).ConvertToRGB32(), \
		rgba.addgrainc(200,200).ConvertToRGB32() \
	), stackvertical(yv12.addgrainc(200,200,constant=true).ConvertToRGB32(), \
		yuy2.addgrainc(200,200,seed=12,constant=true).ConvertToRGB32(), \
		yuy2.addgrainc(200,200,constant=true).ConvertToRGB32(), \
		rgb.addgrainc(200,200,constant=true).ConvertToRGB32(), \
		rgba.addgrainc(200,200,constant=true).ConvertToRGB32() \
	) \
)

