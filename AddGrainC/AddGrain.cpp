#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>
#include <cstring>
#include <vector>

#include "avisynth.h"
#include "emmintrin.h"
#include "smmintrin.h" // sse4

#define VS_RESTRICT __restrict

// max # of noise planes
static constexpr int MAXP = 2;

// offset in pixels of the fake plane MAXP relative to plane MAXP-1
static constexpr int OFFSET_FAKEPLANE = 32;

class AddGrain : public GenericVideoFilter {
  bool _constant;
  int64_t idum;
  int nStride[MAXP], nHeight[MAXP], nSize[MAXP], storedFrames;
  std::vector<uint8_t> pNoiseSeeds;
  std::vector<int16_t> pN[MAXP];
  std::vector<float> pNF[MAXP];
  float _var, _uvar, _hcorr, _vcorr;
  int _seed;

  void setRand(int* plane, int* noiseOffs, const int frameNumber);
  void updateFrame_8_SSE2(uint8_t* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel);
  void updateFrame_16_SSE4(uint16_t* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel);
  template<typename T1>
  void updateFrame(T1* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel);
  template<>
  void updateFrame(float* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int);

public:
  AddGrain(PClip _child, float var, float uvar, float hcorr, float vcorr, int seed, bool constant, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  int __stdcall SetCacheHints(int cachehints, int frame_range)
  {
    return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
  }
};

static inline int64_t fastUniformRandL(int64_t* idum) noexcept {
  return *idum = 1664525LL * (*idum) + 1013904223LL;
}

// very fast & reasonably random
static inline float fastUniformRandF(int64_t* idum) noexcept {
  // work with 32-bit IEEE floating point only!
  fastUniformRandL(idum);
  const uint64_t itemp = 0x3f800000 | (0x007fffff & *idum);
  return *reinterpret_cast<const float*>(&itemp) - 1.f;
}

static inline float gaussianRand(bool* iset, float* gset, int64_t* idum) noexcept {
  float fac, rsq, v1, v2;

  // return saved second
  if (*iset) {
    *iset = false;
    return *gset;
  }

  do {
    v1 = 2.f * fastUniformRandF(idum) - 1.f;
    v2 = 2.f * fastUniformRandF(idum) - 1.f;
    rsq = v1 * v1 + v2 * v2;
  } while (rsq >= 1.f || rsq == 0.f);

  fac = std::sqrt(-2.f * std::log(rsq) / rsq);

  // function generates two values every iteration, so save one for later
  *gset = v1 * fac;
  *iset = true;

  return v2 * fac;
}

static inline float gaussianRand(const float mean, const float variance, bool* iset, float* gset, int64_t* idum) noexcept {
  return (variance == 0.f) ? mean : gaussianRand(iset, gset, idum) * std::sqrt(variance) + mean;
}

// on input, plane is the frame plane index (if applicable, 0 otherwise), and on output, it contains the selected noise plane
void AddGrain::setRand(int* plane, int* noiseOffs, const int frameNumber) {
  if (_constant) {
    // force noise to be identical every frame
    if (*plane >= MAXP) {
      *plane = MAXP - 1;
      *noiseOffs = OFFSET_FAKEPLANE;
    }
  }
  else {
    // pull seed back out, to keep cache happy
    const int seedIndex = frameNumber % storedFrames;
    const int p0 = pNoiseSeeds[seedIndex];

    if (*plane == 0) {
      idum = p0;
    }
    else {
      idum = pNoiseSeeds[(_int64)seedIndex + storedFrames];
      if (*plane == 2) {
        // the trick to needing only 2 planes ^.~
        idum ^= p0;
        (*plane)--;
      }
    }

    // start noise at random qword in top half of noise area
    *noiseOffs = static_cast<int>(fastUniformRandF(&idum) * nSize[*plane] / MAXP) & 0xfffffff8;
  }

  assert(*plane >= 0);
  assert(*plane < MAXP);
  assert(*noiseOffs >= 0);
  assert(*noiseOffs < nSize[*plane]); // minimal check
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
void AddGrain::updateFrame_8_SSE2(uint8_t* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel) {
  const int16_t* pNW = pN[noisePlane].data() + noiseOffs;
  const size_t nstride = nStride[noisePlane];

  // assert(noiseOffs + (nStride[noisePlane] >> 4) * (height - 1) + (stride * 16) <= nSize[noisePlane]);

  const auto sign_mask = _mm_set1_epi8(0x80);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      auto noise_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(pNW + x)); // signed 16
      auto noise_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(pNW + x + 8));
      auto noise = _mm_packs_epi16(noise_lo, noise_hi);
      auto src_signed = _mm_add_epi8(src, sign_mask);
      auto sum = _mm_adds_epi8(noise, src_signed);
      auto result = _mm_add_epi8(sum, sign_mask);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }
    dstp += stride;
    pNW += nstride;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void AddGrain::updateFrame_16_SSE4(uint16_t* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel) {
  const int16_t* pNW = pN[noisePlane].data() + noiseOffs;
  const size_t nstride = nStride[noisePlane];

  // assert(noiseOffs + (nStride[noisePlane] >> 4) * (height - 1) + (stride * 16) <= nSize[noisePlane]);

  auto sign_mask = _mm_set1_epi16(0x8000);

  const int max_pixel_value = 1 << bits_per_pixel;
  auto limit = _mm_set1_epi16(max_pixel_value);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 16) {
      auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      auto noise = _mm_load_si128(reinterpret_cast<const __m128i*>(pNW + x)); // signed 16
      auto src_signed = _mm_add_epi16(src, sign_mask);
      auto sum = _mm_adds_epi16(noise, src_signed);
      auto result = _mm_max_epu16(sum, limit); // for exact 16 bit it's not needed
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }
    dstp += stride;
    pNW += nstride;
  }
}

template<typename T1>
void AddGrain::updateFrame(T1* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int bits_per_pixel) {
  const int upper = 1 << bits_per_pixel;
  const int16_t* pNW = pN[noisePlane].data() + noiseOffs;

  // assert(noiseOffs + (nStride[noisePlane] >> 4) * (height - 1) + (stride * 16) <= nSize[noisePlane]);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if constexpr (sizeof(T1) == 1)
        dstp[x] = (T1)(std::min(std::max(dstp[x] + pNW[x], 0), 255));
      else
        dstp[x] = (T1)(std::min(std::max(dstp[x] + pNW[x], 0), upper));
    }
    dstp += stride;
    pNW += nStride[noisePlane];
  }
}

template<>
void AddGrain::updateFrame(float* VS_RESTRICT dstp, const int width, const int height, const int stride, const int noisePlane, const int noiseOffs, const int) {
  const float* pNW = pNF[noisePlane].data() + noiseOffs;
  assert(noiseOffs + (nStride[noisePlane] >> 4) * (height - 1) + (stride * 16) <= nSize[noisePlane]);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++)
      dstp[x] += pNW[x];

    dstp += stride;
    pNW += nStride[noisePlane];
  }
}

AddGrain::AddGrain(PClip _child, float var, float uvar, float hcorr, float vcorr, int seed, bool constant, IScriptEnvironment* env)
  : GenericVideoFilter(_child), _var(var), _uvar(uvar), _hcorr(hcorr), _vcorr(vcorr), _seed(seed), _constant(constant)
{
  bool iset = false;
  float gset;

  if (_seed < 0)
    _seed = std::time(nullptr); // init random
  idum = _seed;

  int planesNoise = 1;
  nStride[0] = (vi.width + 15) & ~15; // first plane
  nHeight[0] = vi.height;
  if (vi.IsY()) {
    _uvar = 0.f;
  }
  else if (vi.IsRGB()) {
    _uvar = _var;
  }
  else {
    planesNoise = 2;
    nStride[1] = ((vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U)) + 15) & ~15; // second and third plane
    nHeight[1] = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);
  }

  storedFrames = std::min(vi.num_frames, 256);
  pNoiseSeeds.resize((_int64)storedFrames * planesNoise);
  auto pns = pNoiseSeeds.begin();

  float nRep[] = { 2.f, 2.f };
  if (_constant)
    nRep[0] = nRep[1] = 1.f;

  const float pvar[] = { _var, _uvar };
  std::vector<float> lastLine(nStride[0]); // assume plane 0 is the widest one
  const float mean = 0.f;

  const int bits_per_pixel = vi.BitsPerComponent();
  const float noise_scale = bits_per_pixel == 32 ? 1 / 255.f : (float)(1 << (bits_per_pixel - 8));

  for (int plane = 0; plane < planesNoise; plane++)
  {
    int h = static_cast<int>(std::ceil(nHeight[plane] * nRep[plane]));
    if (planesNoise == 2 && plane == 1) {
      // fake plane needs at least one more row, and more if the rows are too small. round to the upper number
      h += (OFFSET_FAKEPLANE + nStride[plane] - 1) / nStride[plane];
    }
    nSize[plane] = nStride[plane] * h;

    // allocate space for noise
    if (vi.BitsPerComponent() != 32)
      pN[plane].resize(nSize[plane]);
    else
      pNF[plane].resize(nSize[plane]);

    for (int x = 0; x < nStride[plane]; x++)
      lastLine[x] = gaussianRand(mean, pvar[plane], &iset, &gset, &idum); // things to vertically smooth against

    for (int y = 0; y < h; y++) {
      if (vi.BitsPerComponent() != 32) {
        auto pNW = pN[plane].begin() + (_int64)nStride[plane] * y;
        float lastr = gaussianRand(mean, pvar[plane], &iset, &gset, &idum); // something to horiz smooth against

        for (int x = 0; x < nStride[plane]; x++) {
          float r = gaussianRand(mean, pvar[plane], &iset, &gset, &idum);

          r = lastr * _hcorr + r * (1.f - _hcorr); // horizontal correlation
          lastr = r;

          r = lastLine[x] * _vcorr + r * (1.f - _vcorr); // vert corr
          lastLine[x] = r;

          *pNW++ = static_cast<int16_t>(std::round(r * noise_scale)); // set noise block
        }
      }
      else {
        auto pNW = pNF[plane].begin() + (_int64)nStride[plane] * y;
        float lastr = gaussianRand(mean, pvar[plane], &iset, &gset, &idum); // something to horiz smooth against

        for (int x = 0; x < nStride[plane]; x++) {
          float r = gaussianRand(mean, pvar[plane], &iset, &gset, &idum);

          r = lastr * _hcorr + r * (1.f - _hcorr); // horizontal correlation
          lastr = r;

          r = lastLine[x] * _vcorr + r * (1.f - _vcorr); // vert corr
          lastLine[x] = r;

          *pNW++ = r * noise_scale; // set noise block
        }
      }
    }

    for (int x = storedFrames; x > 0; x--)
      *pns++ = fastUniformRandL(&idum) & 0xff; // insert seed, to keep cache happy
  }
}


PVideoFrame AddGrain::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  env->MakeWritable(&src);

  int plane;
  int noiseOffs = 0;

  const int bits_per_pixel = vi.BitsPerComponent();

  const bool sse2 = env->GetCPUFlags() & CPUF_SSE2;
  const bool sse41 = env->GetCPUFlags() & CPUF_SSE4_1;

  if (_var > 0.f)
  {
    const int widthY = src->GetRowSize(PLANAR_Y) / vi.ComponentSize();
    const int heightY = src->GetHeight(PLANAR_Y);
    const int strideY = src->GetPitch(PLANAR_Y);
    uint8_t* dstpY = src->GetWritePtr(PLANAR_Y);

    plane = 0;
    int noisePlane = plane;

    setRand(&noisePlane, &noiseOffs, n); // seeds randomness w/ plane & frame

    if (vi.ComponentSize() == 1) {
      if (sse2)
        updateFrame_8_SSE2(dstpY, widthY, heightY, strideY, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint8_t>(dstpY, widthY, heightY, strideY, noisePlane, noiseOffs, bits_per_pixel);
    }
    else if (vi.ComponentSize() == 2) {
      if (sse41)
        updateFrame_16_SSE4(reinterpret_cast<uint16_t*>(dstpY), widthY, heightY, strideY / 2, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint16_t>(reinterpret_cast<uint16_t*>(dstpY), widthY, heightY, strideY / 2, noisePlane, noiseOffs, bits_per_pixel);
    }
    else
      updateFrame<float>(reinterpret_cast<float*>(dstpY), widthY, heightY, strideY / 4, noisePlane, noiseOffs, bits_per_pixel);
  }

  if (_uvar > 0.f)
  {
    const int widthUV = src->GetRowSize(PLANAR_U) / vi.ComponentSize();
    const int heightUV = src->GetHeight(PLANAR_U);
    const int strideUV = src->GetPitch(PLANAR_U);
    uint8_t* dstpU = src->GetWritePtr(PLANAR_U);
    uint8_t* dstpV = src->GetWritePtr(PLANAR_V);

    plane = 1;
    int noisePlane = (vi.IsRGB()) ? 0 : plane;

    setRand(&noisePlane, &noiseOffs, n); // seeds randomness w/ plane & frame

    if (vi.ComponentSize() == 1) {
      if (sse2)
        updateFrame_8_SSE2(dstpU, widthUV, heightUV, strideUV, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint8_t>(dstpU, widthUV, heightUV, strideUV, noisePlane, noiseOffs, bits_per_pixel);
    }
    else if (vi.ComponentSize() == 2) {
      if (sse41)
        updateFrame_16_SSE4(reinterpret_cast<uint16_t*>(dstpU), widthUV, heightUV, strideUV / 2, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint16_t>(reinterpret_cast<uint16_t*>(dstpU), widthUV, heightUV, strideUV / 2, noisePlane, noiseOffs, bits_per_pixel);
    }
    else
      updateFrame<float>(reinterpret_cast<float*>(dstpU), widthUV, heightUV, strideUV / 4, noisePlane, noiseOffs, bits_per_pixel);

    plane = 2;
    noisePlane = (vi.IsRGB()) ? 0 : plane;

    setRand(&noisePlane, &noiseOffs, n); // seeds randomness w/ plane & frame

    if (vi.ComponentSize() == 1) {
      if (sse2)
        updateFrame_8_SSE2(dstpV, widthUV, heightUV, strideUV, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint8_t>(dstpV, widthUV, heightUV, strideUV, noisePlane, noiseOffs, bits_per_pixel);
    }
    else if (vi.ComponentSize() == 2) {
      if (sse41)
        updateFrame_16_SSE4(reinterpret_cast<uint16_t*>(dstpV), widthUV, heightUV, strideUV / 2, noisePlane, noiseOffs, bits_per_pixel);
      else
        updateFrame<uint16_t>(reinterpret_cast<uint16_t*>(dstpV), widthUV, heightUV, strideUV / 2, noisePlane, noiseOffs, bits_per_pixel);
    }
    else
      updateFrame<float>(reinterpret_cast<float*>(dstpV), widthUV, heightUV, strideUV / 4, noisePlane, noiseOffs, bits_per_pixel);
  }

  return src;

}

AVSValue __cdecl Create_AddGrain(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  if ((args[3].AsFloat(0)) < 0.f || (args[3].AsFloat(0)) > 1.f || (args[4].AsFloat(0)) < 0.f || (args[4].AsFloat(0)) > 1.f) {
    env->ThrowError("AddGrain: hcorr and vcorr must be between 0.0 and 1.0 (inclusive)");
  }

  return new AddGrain(args[0].AsClip(), args[1].AsFloat(1), args[2].AsFloat(0), args[3].AsFloat(0), args[4].AsFloat(0), args[5].AsInt(-1), args[6].AsBool(false), env);
}

AVSValue __cdecl Create_AddGrainC(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  if ((args[3].AsFloat(0)) < 0.f || (args[3].AsFloat(0)) > 1.f || (args[4].AsFloat(0)) < 0.f || (args[4].AsFloat(0)) > 1.f) {
    env->ThrowError("AddGrain: hcorr and vcorr must be between 0.0 and 1.0 (inclusive)");
  }

  return new AddGrain(args[0].AsClip(), args[1].AsFloat(1), args[2].AsFloat(0), args[3].AsFloat(0), args[4].AsFloat(0), args[5].AsInt(-1), args[6].AsBool(false), env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
  AVS_linkage = vectors;

  env->AddFunction("AddGrain", "c[var]f[uvar]f[hcorr]f[vcorr]f[seed]i[constant]b", Create_AddGrain, NULL);
  env->AddFunction("AddGrainC", "c[var]f[uvar]f[hcorr]f[vcorr]f[seed]i[constant]b", Create_AddGrainC, NULL);
  return "`AddGrainC' Add some correlated color gaussian noise";

}