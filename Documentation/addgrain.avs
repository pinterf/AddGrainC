ColorBars(width=640, height=100, pixel_type="yuv420p8")
orig = last

bits = 32 # set to bit depth to test
c1 = orig.ConvertBits(bits)
c1 = c1.addgrain_addgrain(var = 100)
# uncomment for speed test
#return c1
#__END__

c2 = orig.addgrainc_addgrain(var = 100)

c0 = orig.ScriptClip("""SubTitle("original AverageLuma =" + String(AverageLuma() / ( Bitshl(1, BitsPerComponent() - 8))))""")
IF(c1.IsVideoFloat()) {
  c1 =   c1.ScriptClip("""SubTitle("addgrain AverageLuma =" + String(AverageLuma() * 255.0 ))""")
} else {
  c1 =   c1.ScriptClip("""SubTitle("addgrain AverageLuma =" + String(AverageLuma() / ( Bitshl(1, BitsPerComponent() - 8))))""")
}
c2 =   c2.ScriptClip("""SubTitle("addgrainc AverageLuma=" + String(AverageLuma() / ( Bitshl(1, BitsPerComponent() - 8))))""")

StackVertical(c0,c1.ConvertBits(8), c2.ConvertBits(8), Diff(c1, c2))
return last

Function Diff(clip src1, clip src2)
{
  return Subtract(src1.ConvertBits(8),src2.ConvertBits(8)).Levels(120, 1, 255-120, 0, 255, coring=false)
}

