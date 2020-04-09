// Copyright (c) 2002 Tom Barry.  All rights reserved.
//		trbarry@trbarry.com
//	modified by Foxyshadis
//		foxyshadis@hotmail.com
//	modified by Firesledge
//		http://ldesoras.free.fr
//	modified by LaTo INV.
//		http://forum.doom9.org/member.php?u=131032
// Requires Avisynth source code to compile for Avisynth
// Avisynth Copyright 2000 Ben Rudiak-Gould.
//      http://www.math.berkeley.edu/~benrg/avisynth.html
/////////////////////////////////////////////////////////////////////////////
//
//	This file is subject to the terms of the GNU General Public License as
//	published by the Free Software Foundation.  A copy of this license is
//	included with this software distribution in the file COPYING.  If you
//	do not have a copy, you may obtain a copy by writing to the Free
//	Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
//
//	This software is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details
//  
//  Also, this program is "Philanthropy-Ware".  That is, if you like it and 
//  feel the need to reward or inspire the author then please feel free (but
//  not obligated) to consider joining or donating to the Electronic Frontier
//  Foundation. This will help keep cyber space free of barbed wire and bullsh*t.  
//
/////////////////////////////////////////////////////////////////////////////
// Change Log
//
// Date          Version  Developer      Changes
//

// 07 May 2003   1.0.0.0  Tom Barry      New Release
// 01 Jun 2006   1.1.0.0  Foxyshadis     Chroma noise, constant seed
// 06 Jun 2006   1.2.0.0  Foxyshadis     Supports YUY2, RGB. Fix cache mess.
// 10 Jun 2006   1.3.0.0  Foxyshadis     Crashfix, noisegen optimization
// 11 Nov 2006   1.4.0.0  Foxyshadis     Constant replaces seed, seed repeatable
// 07 May 2010   1.5.0.0  Foxyshadis     Limit the initial seed generation to fix memory issues.
// 13 May 2010   1.5.1.0  Firesledge     The source code compiles on Visual C++ versions older than 2008
// 26 Oct 2011   1.5.2.0  Firesledge     Removed the SSE2 requirement.
// 26 Oct 2011   1.5.3.0  Firesledge     Fixed coloring and bluring in RGB24 mode.
// 27 Oct 2011   1.5.4.0  Firesledge     Fixed bad pixels on the last line in YV12 mode when constant=true,
//                                       fixed potential problems with frame width > 4096 pixels
//                                       and fixed several other minor things.
// 28 Oct 2011   1.6.0.0  LaTo INV.      Added SSE2 code (50% faster than MMX).
// 29 Oct 2011   1.6.1.0  LaTo INV.      Automatic switch to MMX if SSE2 is not supported by the CPU.
// 16 Aug 2012   1.7.0.0  Firesledge     Supports Y8, YV16, YV24 and YV411 colorspaces.
//
/////////////////////////////////////////////////////////////////////////////
// CVS Log
//
//
/////////////////////////////////////////////////////////////////////////////

#if defined (_WIN64) || defined (__64BIT__) || defined (__amd64__) || defined (__x86_64__)
	#define	AddGrainC_ARCH_BITS	64
#else
	#define	AddGrainC_ARCH_BITS	32
#endif

#ifdef DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#include "avisynth.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <ctime>

#if defined (_MSC_VER) && _MSC_VER < 1500
	typedef	unsigned char	uint8_t;
	typedef	signed char	int8_t;
#else
	#include <stdint.h>
#endif

#include	<emmintrin.h>

#define ALIGN8(x) ((((x) + 7) >>3 ) << 3)
#define ALIGN16(x) ((((x) + 15) >>4 ) << 4)
#define ALIGN24(x) ((((x) + 23) / 24) * 24)

// max # of noise planes, try 3 if you want to rewrite the yv12 chroma routines
#define MAXP (2)

// Offset in pixels of the fake plane MAXP relative to plane MAXP-1.
#define OFFSET_FAKEPLANE	(32)

class AddGrain : public GenericVideoFilter {
	int  Max_Frames;

	int planes_noise;
	int planes_colorspace;
	float variance, uvariance;
	float Hcorr, Vcorr;
	int seed,storedframes;
	bool iset, constant, sse2;
	float gset;
	long idum;
	std::vector <uint8_t> pNoiseSeeds;
	// increase this to lessen inter-frame noise coherence and increase memory
	float NRep[MAXP];
	std::vector <int8_t> pN[MAXP];
	int NPitch[MAXP], NHeight[MAXP], NSize[MAXP];

	PVideoFrame GetFramePlanar(int inFrame, IScriptEnvironment* env);
	PVideoFrame GetFrameInterleaved(int inFrame, IScriptEnvironment* env);
	BOOL UpdateFrame(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs);
#if (AddGrainC_ARCH_BITS == 32)
	BOOL UpdateFrame_MMX(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs);
#endif
	BOOL UpdateFrame_SSE2(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs);

	int	get_width (const ::PVideoFrame &frame_sptr, int plane_id) const;
	int	get_height (const ::PVideoFrame &frame_sptr, int plane_id) const;

	float FastUniformRandF();
	long FastUniformRandL();
	float GaussianRand();
	float GaussianRand(float mean, float variance);
	void SetRand(int &plane, int &NoiseOffs, int frame_number );

	static int	yuv_plane_index_to_id (int index);

public:

	AddGrain(PClip, float, float, float, float, int, bool, bool, IScriptEnvironment*);
	PVideoFrame __stdcall GetFrame(int inFrame, IScriptEnvironment* env);
	~AddGrain() 
	{
		// Nothing now.
	}
};

AddGrain::AddGrain(PClip _child, float _variance, float _Hcorr, float _Vcorr, float _uvariance, 
		int _seed, bool _constant, bool _sse2, IScriptEnvironment* env)
:	GenericVideoFilter(_child)
,	variance(_variance)
,	Hcorr(_Hcorr)
,	Vcorr(_Vcorr)
,	uvariance(_uvariance)
,	seed(_seed)
,	constant(_constant)
,	sse2(_sse2)
{
	if ( sse2 && ((env->GetCPUFlags() & CPUF_SSE2) == 0) ) 
	{
		sse2 = false; // Turn off SSE2 if CPU doesn't support it 
	}

	if ( !sse2 && ((env->GetCPUFlags() & CPUF_MMX) == 0) ) 
	{
		env->ThrowError("AddGrain: Only MMX & SSE2 CPU's currently supported.");
	}

    if (Hcorr < 0.0 || Hcorr > 1.0 || Vcorr < 0.0 || Vcorr > 1.0)
	{
		env->ThrowError("AddGrain: HCorr & VCorr must be 0 <= x <= 1.0");
	}

	child->SetCacheHints(CACHE_NOTHING, 0);
	iset = false;
	if(seed<0)
		seed = (unsigned long)time(NULL);		// init random
	idum = seed;

	// set up requirements for different colorspaces
	planes_colorspace = 0;	// Valid only for planar modes
	if (vi.IsPlanar () && vi.IsYUV ())
	{
		planes_noise = 1;
		planes_colorspace = vi.IsY8 () ? 1 : 3;
		NPitch[0] = ALIGN8(vi.width);		 // luma
		NHeight[0] = vi.height;
		if (planes_colorspace == 1)
		{
			uvariance = 0;	// Prevents to render the U and V planes in GetFramePlanar()
		}
		else
		{
			planes_noise = 2;
			const int		sshl2 = vi.GetPlaneWidthSubsampling (PLANAR_U);
			const int		ssvl2 = vi.GetPlaneHeightSubsampling (PLANAR_U);
			NPitch[1] = ALIGN8(vi.width >> sshl2);	 // chroma
			NHeight[1] = vi.height >> ssvl2;
		}
	} else if(vi.IsYUY2()) {
		planes_noise = 2;
		NPitch[0] = NPitch[1] = ALIGN8(vi.width * 2);
		NHeight[0] = NHeight[1] = vi.height;
	} else if(vi.IsRGB()) {
		planes_noise = 1;
		const int	pitch_tmp = vi.width * vi.BytesFromPixels(1);
		if (vi.IsRGB24())
		{
			NPitch[0] = ALIGN24 (pitch_tmp);
		}
		else
		{
			NPitch[0] = ALIGN8 (pitch_tmp);
		}
		NHeight[0] = vi.height;
	} else {
		env->ThrowError("AddGrain: Unsupported colorspace!");
	}
	storedframes = min(vi.num_frames,256);
	long NoiseSize = storedframes * planes_noise;
	pNoiseSeeds.resize (NoiseSize);
	std::vector <uint8_t>::iterator pns = pNoiseSeeds.begin (); 
	NRep[0] = NRep[1] = 2;
	if(constant) {
		//seed = seed % NoiseSize;
		NRep[0] = 1;
		NRep[1] = 1;
	}

	float pvar[] = { variance, uvariance };
	std::vector <float> LastLine (NPitch [0]);	// Assumes plane 0 is the widest one.
	float mean = 0;
	for(int plane=0; plane < planes_noise; plane++)	{
		int h = int (ceil (NHeight[plane] * NRep[plane]));
		if (planes_noise == 2 && plane == 1)
		{
			// Fake plane needs at least one more row, and more if
			// the rows are too small. Rounds to the upper number.
			h += (OFFSET_FAKEPLANE + NPitch[plane] - 1) / NPitch[plane];
		}
		NSize[plane] = NPitch[plane] * h;
		// allocate space for noise
		pN[plane].resize (NSize[plane]);
		int x;
		for (x=0 ; x < NPitch[plane]; x++) {
			LastLine[x] = GaussianRand(mean, pvar[plane]);		// things to vertically smooth against
		}
		for (int y=0; y < h; y++) {
			std::vector <int8_t>::iterator pNW =
				pN [plane].begin () + y * NPitch [plane];
			float Lastr = GaussianRand(mean, pvar[plane]);   // something to horiz smooth against
			for (x = 0; x < NPitch[plane]; x++) {
				float r = GaussianRand(mean, pvar[plane]);
				r = Lastr* Hcorr + r * (1-Hcorr);  // horizontal correlation
				Lastr = r;
				r = LastLine[x] * Vcorr + r * (1-Vcorr); // vert corr
				LastLine[x] = r;

				// set noise block
				const int8_t	r8 = int8_t (floor (r + 0.5));	// Round to nearest
				if(vi.IsPlanar () && vi.IsYUV ()) {
					*pNW++ = r8 ;
				} else if(vi.IsYUY2()) {
					*pNW++ = (x++ + plane) % 2 == 0 ? r8 : 0;
					*pNW++ = (x   + plane) % 2 == 0 ? r8 : 0;
					// alternating pixels: plane 0 sets y0y0, plane 1 sets 0u0v
				} else if(vi.IsRGB()) {
					*pNW++ = r8 ;		// all channels equal, no chroma grain
					*pNW++ = r8 ; x++;
					*pNW++ = r8 ; x++;
					if(vi.IsRGB32()) {
						*pNW++ = 0;  x++;		// alpha channel unused
					}
				}
			}
		}
		for (x=storedframes; x > 0; x--) {
			// insert seed, to keep cache happy
			*pns++ = int8_t (FastUniformRandL() & 0xff);
		}
	}
}



PVideoFrame __stdcall AddGrain::GetFrame(int inFrame, IScriptEnvironment* env) {
	if(vi.IsPlanar()) {
		return GetFramePlanar(inFrame,env);
	} else {
		return GetFrameInterleaved(inFrame,env);
	}
}



PVideoFrame AddGrain::GetFramePlanar(int inFrame, IScriptEnvironment* env) {
	PVideoFrame src = child->GetFrame(inFrame, env);
	env->MakeWritable(&src);

	int plane;
	int NoiseOffs;

	if(variance > 0.0f) {
		BYTE* dstpY      = src->GetWritePtr(PLANAR_Y);
		int   dst_pitchY = src->GetPitch(PLANAR_Y);
		int   row_sizeY  = get_width (src, PLANAR_Y);   // Could also be PLANAR_Y_ALIGNED which would return a mod16 row_size
		int   heightY    = get_height (src, PLANAR_Y);

		plane = 0;
		SetRand (plane, NoiseOffs, inFrame);	// seeds randomness w/ plane & frame
		UpdateFrame(dst_pitchY,	 row_sizeY,  dstpY, heightY, plane, NoiseOffs);
	}

	if(uvariance > 0.0f) {
		BYTE* dstpV       = src->GetWritePtr(PLANAR_V);   
		BYTE* dstpU       = src->GetWritePtr(PLANAR_U);
		int   dst_pitchUV = src->GetPitch(PLANAR_U);
		int   row_sizeUV  = get_width (src, PLANAR_U);  // Could also be PLANAR_U_ALIGNED which would return a mod8 row_size
		int   heightUV    = get_height (src, PLANAR_U);

		plane = 1;
		SetRand (plane, NoiseOffs, inFrame);
		UpdateFrame(dst_pitchUV, row_sizeUV, dstpU, heightUV, plane, NoiseOffs);

		plane = 2;
		SetRand (plane, NoiseOffs, inFrame);
		UpdateFrame(dst_pitchUV, row_sizeUV, dstpV, heightUV, plane, NoiseOffs);
	}

	return src;
}



PVideoFrame  AddGrain::GetFrameInterleaved(int inFrame, IScriptEnvironment* env) {
	PVideoFrame src = child->GetFrame(inFrame, env);
	env->MakeWritable(&src);
	BYTE* dstp;
	int dst_pitch;
	int row_size;
	int height;

	dstp = src->GetWritePtr();
	dst_pitch = src->GetPitch();
	row_size = src->GetRowSize();
	height = src->GetHeight();

	int plane;
	int NoiseOffs;
	if(variance > 0.0f) {
		plane = 0;
		SetRand (plane, NoiseOffs, inFrame);
		UpdateFrame(dst_pitch, row_size, dstp, height, plane, NoiseOffs);
	}

	if(uvariance > 0.0f && vi.IsYUY2()) {
		plane = 1;
		SetRand (plane, NoiseOffs, inFrame);
		UpdateFrame(dst_pitch, row_size, dstp, height, plane, NoiseOffs);
	}

	return src;
}



// todo: parallelize
BOOL	AddGrain::UpdateFrame(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs)	
{
	BOOL value;

#if (AddGrainC_ARCH_BITS == 32)
	if (sse2)
	{
#endif
		value = UpdateFrame_SSE2(dst_pit, row_size, dstp, FldHeight, plane, NoiseOffs);
#if (AddGrainC_ARCH_BITS == 32)
	}
	else
	{
		value = UpdateFrame_MMX(dst_pit, row_size, dstp, FldHeight, plane, NoiseOffs);
	}
#endif

	return value;
}



#if (AddGrainC_ARCH_BITS == 32)

BOOL	AddGrain::UpdateFrame_MMX(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs)	
{
	__int64 All128 = 0x8080808080808080;

	int ct = (row_size+7) >> 3;
	int8_t* pNW2 = &(pN [plane] [NoiseOffs]);
	int NoisePitch2 = NPitch[plane];
	assert (NoiseOffs + NoisePitch2 * (FldHeight - 1) + ct * 8 <= NSize [plane]);

	__asm
	{
		emms

		mov      edi, dstp               // ptr to src and dst
		mov      esi, pNW2               // our starting loc in noise
		mov      edx, FldHeight          // ctr for num lines
		movq     mm7, All128

		align	16
	NoisyLineLoop:
		xor      ebx, ebx                // offset into lines
		mov      ecx, ct
		
		align	16
	NoisyCharLoop:
		movq     mm0, qword ptr[edi+ebx]	
		psubb    mm0, mm7                // turn into signed number
		paddsb   mm0, qword ptr[esi+ebx] // add in our noise, saturate overflow
		paddb    mm0, mm7                // turn back into unsigned number
		movq     qword ptr[edi+ebx], mm0 // and put it back in place
		add      ebx, 8                  // bump offset to next qword
		loop     NoisyCharLoop           // loop for next byte

		add      edi, dst_pit
		add      esi, NoisePitch2
		dec      edx
		jnz      NoisyLineLoop           // loop for next line
		
		emms
	}

	return 0;
}

#endif



BOOL	AddGrain::UpdateFrame_SSE2(int dst_pit, int row_size, BYTE* dstp, int FldHeight, int plane, int NoiseOffs)	
{
	// Makes sure dstp is aligned on 16 bytes.
	const ptrdiff_t ofs = reinterpret_cast <ptrdiff_t> (dstp) & 15;
	dstp     -= ofs;
	row_size += int (ofs);

	int ct = (row_size+15) >> 4;
	int8_t* pNW2 = &(pN [plane] [NoiseOffs]);
	int NoisePitch2 = NPitch[plane];
	assert (NoiseOffs + NoisePitch2 * (FldHeight - 1) + ct * 16 <= NSize [plane]);

#if (AddGrainC_ARCH_BITS == 32)

	uint8_t All128[16] = { 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 
						   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 };

	__asm
	{
		mov      edi,  dstp              // ptr to src and dst
		mov      esi,  pNW2              // our starting loc in noise
		mov      edx,  FldHeight         // ctr for num lines
		movdqu   xmm7, All128

	NoisyLineLoop:
		xor      ebx, ebx                // offset into lines
		mov      ecx, ct
		
	NoisyCharLoop:
		movdqa   xmm0, [edi+ebx]
		psubb    xmm0, xmm7              // turn into signed number
		movdqu   xmm1, [esi+ebx]
		paddsb   xmm0, xmm1              // add in our noise, saturate overflow
		paddb    xmm0, xmm7              // turn back into unsigned number
		movdqa   [edi+ebx], xmm0         // and put it back in place
		add      ebx,  16                // bump offset to next dqword
		loop     NoisyCharLoop           // loop for next byte

		add      edi, dst_pit
		add      esi, NoisePitch2
		dec      edx
		jnz      NoisyLineLoop           // loop for next line
	}

#else

	const __m128i sign = _mm_set1_epi8 (-0x80);
	for (int y = 0; y < FldHeight; ++y)
	{
		for (int x = 0; x < ct; ++x)
		{
			__m128i        val = _mm_load_si128 (reinterpret_cast <__m128i *> (dstp) + x);
			const __m128i  nz  = _mm_loadu_si128 (reinterpret_cast <__m128i *> (pNW2) + x);
			val = _mm_xor_si128 (val, sign);
			val = _mm_adds_epi8 (val, nz);
			val = _mm_xor_si128 (val, sign);
			_mm_store_si128 (reinterpret_cast <__m128i *> (dstp) + x, val);
		}

		dstp += dst_pit;
		pNW2 += NoisePitch2;
	}

#endif

	return 0;
}



int	AddGrain::get_width (const ::PVideoFrame &frame_sptr, int plane_id) const
{
	assert (&frame_sptr != 0);
	assert (frame_sptr != 0);

	int				width = 0;
	if (plane_id == PLANAR_U || plane_id == PLANAR_V)
	{
		width = frame_sptr->GetRowSize (PLANAR_Y);
		const int		subspl = vi.GetPlaneWidthSubsampling (plane_id);
		width >>= subspl;
	}
	else
	{
		width = frame_sptr->GetRowSize (plane_id);
	}

	return (width);
}



int	AddGrain::get_height (const ::PVideoFrame &frame_sptr, int plane_id) const
{
	assert (&frame_sptr != 0);
	assert (frame_sptr != 0);

	int				height = 0;
	if (plane_id == PLANAR_U || plane_id == PLANAR_V)
	{
		height = frame_sptr->GetHeight (PLANAR_Y);
		const int		subspl = vi.GetPlaneHeightSubsampling (plane_id);
		height >>= subspl;
	}
	else
	{
		height = frame_sptr->GetHeight (plane_id);
	}

	return (height);
}



// very fast & reasonably random
inline float AddGrain::FastUniformRandF() {
	// Works with 32-bit IEEE floating point only!

	FastUniformRandL();
	unsigned long itemp = 0x3f800000 | (0x007fffff & idum);
	return (*(float*)&itemp) - 1.0f;
}

inline long AddGrain::FastUniformRandL() {
	return idum = 1664525L * idum + 1013904223L;
}



float AddGrain::GaussianRand()
{
	float fac, rsq, v1, v2;

	// return saved second
	if(iset)
	{
		iset = false;
		return gset;
	}

	do
	{
		v1 = 2.0f * FastUniformRandF() - 1.0f;
		v2 = 2.0f * FastUniformRandF() - 1.0f;
		rsq = v1 * v1 + v2 * v2;
	} while(rsq >= 1.0f || 0.0f == rsq);

	fac = (float)sqrt(-2.0f * (float)log(rsq) / rsq);

	// function generates two values every iteration, so save one for later
	gset = v1 * fac;
	iset = true;

	return v2 * fac;
}

float AddGrain::GaussianRand(float mean, float variance)
{
	if(variance == 0.0f) return mean;
	return GaussianRand() * (float)sqrt(variance) + mean;
}



// On input, plane is the frame plane index (if applicable, 0 otherwise),
// and on output, it contains the selected noise plane.
void AddGrain::SetRand(int &plane, int &NoiseOffs, int frame_number = -1) {
	NoiseOffs = 0;
	if(constant)
	{
		// force noise to be identical every frame
		if(plane>=MAXP) {
			plane = MAXP - 1;
			NoiseOffs = OFFSET_FAKEPLANE;
		}
	}
	else
	{
		if(frame_number >= 0) {
			// pull seed back out, to keep cache happy
			const int		seed_index = frame_number % storedframes;
			int p0 = pNoiseSeeds [seed_index];
			if (plane == 0)
			{
				idum = p0;
			}
			else
			{
				idum = pNoiseSeeds [seed_index + storedframes];
				if (plane == 2)
				{
					// the trick to needing only 2 planes for yv12 ^.~
					idum ^= p0;
					plane--;
				}
			}
		}
		// start noise at random qword in top half of noise area
		const int		raw_offset = int (FastUniformRandF() * NSize[plane]/MAXP);
		if (vi.IsRGB24 ())
		{
			// Ensures that color triplets keep their alignment too.
			const int		alig = 8 * 3;
			NoiseOffs = raw_offset - raw_offset % alig;
		}
		else
		{
			NoiseOffs = raw_offset & 0xfffffff8;
		}
	}
	assert (plane >= 0);
	assert (plane < MAXP);
	assert (NoiseOffs >= 0);
	assert (NoiseOffs < NSize[plane]);	// Minimal check
}



#pragma warning(disable:4244) // disable conversion from double to float message

AVSValue __cdecl Create_AddGrain(AVSValue args, void* user_data,
IScriptEnvironment* env) 
{
	enum { CLIP, VARIANCE, HCORR, VCORR, UVARIANCE, SEED, CONSTANT, SSE2 };
	return new AddGrain(args[CLIP].AsClip(), args[VARIANCE].AsFloat(1.0), args[HCORR].AsFloat(0.0),
		args[VCORR].AsFloat(0.0), args[UVARIANCE].AsFloat(0.0), args[SEED].AsInt(-1), args[CONSTANT].AsBool(false), args[SSE2].AsBool(true),  env);
}

AVSValue __cdecl Create_AddGrainC(AVSValue args, void* user_data,
IScriptEnvironment* env) 
{
	enum { CLIP, VARIANCE, UVARIANCE, HCORR, VCORR, SEED, CONSTANT, SSE2 };
	return new AddGrain(args[CLIP].AsClip(), args[VARIANCE].AsFloat(1.0), args[HCORR].AsFloat(0.0),
		args[VCORR].AsFloat(0.0), args[UVARIANCE].AsFloat(0.0), args[SEED].AsInt(-1), args[CONSTANT].AsBool(false), args[SSE2].AsBool(true),  env);
}

extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit2(IScriptEnvironment* env) {
			_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    env->AddFunction("AddGrain", "c[var]f[hcorr]f[vcorr]f[uvar]f[seed]i[constant]b[sse2]b", Create_AddGrain, 0);
    env->AddFunction("AddGrainC", "c[var]f[uvar]f[hcorr]f[vcorr]f[seed]i[constant]b[sse2]b", Create_AddGrainC, 0);
    return "`AddGrainC' Add some correlated color gaussian noise";
}

