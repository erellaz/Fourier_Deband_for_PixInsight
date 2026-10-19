// ----------------------------------------------------------------------------
// FourierDeband.js  -  PixInsight JavaScript Runtime (PJSR) script
// ----------------------------------------------------------------------------
// Fourier-domain removal of CMOS row/column banding and other periodic noise.
//
//  * Horizontal bands (row-to-row offsets)   -> notch the fx = 0 axis
//  * Vertical bands (column-to-column offsets) -> notch the fy = 0 axis
//    - Gaussian notch across the axis (width in frequency bins)
//    - raised-cosine protection of low frequencies (large-scale signal)
//  * Periodic pattern noise: automatic detection of off-axis spectral spikes
//    against a robust local background of log|F|, magnitude pulled back to
//    that background (no zero-holes, no ringing)
//  * Checkerboard / Nyquist pattern (debayer, 2x drizzle, dual readout)
//  * Optional adaptive amplitude matching: smoothly varying gain per band
//    component, fitted on unprotected pixels with shrinkage towards 1
//
//  Artifact control:
//  * Moisan periodic+smooth decomposition (J. Math. Imaging Vis. 39, 2011):
//    removes the image-edge discontinuity "cross" that otherwise sits exactly
//    on the spectral axes being notched.
//  * Noise is estimated on a copy where stars, halos, spikes and saturated cores
//    are masked (iterated masked background, hysteresis mask) and refilled (POCS),
//    then subtracted from the original: bright stars leave no streaks.
//
//  Self-contained: pure-JS mixed-radix / Bluestein FFT, any image size.
//  Memory ~ 22 bytes/pixel per channel (~26 with adaptive matching);
//  a 9576 x 6388 frame needs about 1.4-1.6 GB.
//
//  Documentation: FourierDeband_how_it_works.docx (algorithm, equations, validation).
//  Author: BlackRig (erellaz).  License: MIT.
//
//  Modes: the active image, or a batch of files (e.g. calibrated subs before
//  registration); batch output keeps FITS keywords, properties and astrometry.
//
//  Install: Script > Feature Scripts... > Add (folder containing this file),
//  then Script > BlackRig > FourierDeband.  Or: Script > Execute Script File.
// ----------------------------------------------------------------------------

#feature-id    FourierDeband : BlackRig > FourierDeband
#feature-info  Removes CMOS banding (horizontal and/or vertical) and periodic \
               pattern noise by notch filtering in the Fourier domain, with \
               star protection and edge-artifact suppression.

#include <pjsr/Sizer.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/StdButton.jsh>
#include <pjsr/StdIcon.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/NumericControl.jsh>
#include <pjsr/DataType.jsh>
#include <pjsr/StdCursor.jsh>

#define FD_VERSION "2.0.1"

// ==== CORE BEGIN ====
// Pure ECMAScript 5 numerical core (no PJSR dependencies) so it can be
// unit-tested outside PixInsight.

// ---------------------------------------------------------------------
// 1-D complex FFT: mixed radix (4, 2, generic small primes), KISS-FFT
// structure; Bluestein chirp-z for lengths with a prime factor > 31.
// Only the forward transform is implemented; inverse = conj(F(conj(x))).
// ---------------------------------------------------------------------
function FD_factorize(n)
{
   var f = [], p = 4, fl = Math.floor(Math.sqrt(n));
   do
   {
      while (n % p)
      {
         switch (p)
         {
            case 4: p = 2; break;
            case 2: p = 3; break;
            default: p += 2; break;
         }
         if (p > fl) p = n;
      }
      n /= p;
      f.push(p);
      f.push(n);
   }
   while (n > 1);
   return f;
}

function FD_FFTPlan(n)
{
   this.n = n;
   this.bluestein = false;
   if (n < 2)
   {
      this.factors = [1, 1];
      return;
   }
   var f = FD_factorize(n), maxp = 1, i;
   for (i = 0; i < f.length; i += 2)
      if (f[i] > maxp) maxp = f[i];
   if (maxp > 31)
   {
      // Bluestein
      this.bluestein = true;
      var m = 1;
      while (m < 2 * n - 1) m *= 2;
      this.m = m;
      this.sub = new FD_FFTPlan(m);
      this.wRe = new Float64Array(n);
      this.wIm = new Float64Array(n);
      var twoN = 2 * n;
      for (i = 0; i < n; ++i)
      {
         var kk = (i * i) % twoN;           // exact for n < ~6.7e7
         var a = Math.PI * kk / n;
         this.wRe[i] = Math.cos(a);
         this.wIm[i] = -Math.sin(a);        // w_k = exp(-i*pi*k^2/n)
      }
      var bRe = new Float64Array(m), bIm = new Float64Array(m);
      bRe[0] = this.wRe[0]; bIm[0] = -this.wIm[0];
      for (i = 1; i < n; ++i)
      {
         bRe[i] = bRe[m - i] = this.wRe[i];
         bIm[i] = bIm[m - i] = -this.wIm[i];   // conj(w)
      }
      this.BRe = new Float64Array(m);
      this.BIm = new Float64Array(m);
      this.sub.transform(bRe, bIm, this.BRe, this.BIm);
      this.aRe = new Float64Array(m); this.aIm = new Float64Array(m);
      this.cRe = new Float64Array(m); this.cIm = new Float64Array(m);
      return;
   }
   this.factors = f;
   this.twRe = new Float64Array(n);
   this.twIm = new Float64Array(n);
   for (i = 0; i < n; ++i)
   {
      var ang = -2 * Math.PI * i / n;
      this.twRe[i] = Math.cos(ang);
      this.twIm[i] = Math.sin(ang);
   }
   this.scRe = new Float64Array(maxp);
   this.scIm = new Float64Array(maxp);
}

FD_FFTPlan.prototype.transform = function(iRe, iIm, oRe, oIm)
{
   var n = this.n;
   if (n < 2) { oRe[0] = iRe[0]; oIm[0] = iIm[0]; return; }
   if (!this.bluestein)
   {
      this.work(oRe, oIm, 0, iRe, iIm, 0, 1, 0);
      return;
   }
   var m = this.m, i, wr, wi, xr, xi;
   var aRe = this.aRe, aIm = this.aIm, cRe = this.cRe, cIm = this.cIm;
   for (i = 0; i < n; ++i)
   {
      wr = this.wRe[i]; wi = this.wIm[i]; xr = iRe[i]; xi = iIm[i];
      aRe[i] = xr * wr - xi * wi;
      aIm[i] = xr * wi + xi * wr;
   }
   for (i = n; i < m; ++i) { aRe[i] = 0; aIm[i] = 0; }
   this.sub.transform(aRe, aIm, cRe, cIm);
   // multiply by B, then inverse via conj trick
   var BRe = this.BRe, BIm = this.BIm;
   for (i = 0; i < m; ++i)
   {
      xr = cRe[i]; xi = cIm[i];
      aRe[i] = xr * BRe[i] - xi * BIm[i];
      aIm[i] = -(xr * BIm[i] + xi * BRe[i]);     // conjugate
   }
   this.sub.transform(aRe, aIm, cRe, cIm);
   var inv = 1 / m;
   for (i = 0; i < n; ++i)
   {
      xr = cRe[i] * inv; xi = -cIm[i] * inv;     // conj back, normalize
      wr = this.wRe[i]; wi = this.wIm[i];
      oRe[i] = xr * wr - xi * wi;
      oIm[i] = xr * wi + xi * wr;
   }
};

FD_FFTPlan.prototype.work = function(oRe, oIm, oo, iRe, iIm, fi, fstride, fidx)
{
   var p = this.factors[fidx], m = this.factors[fidx + 1];
   var oend = oo + p * m, o;
   if (m == 1)
   {
      for (o = oo; o < oend; ++o)
      {
         oRe[o] = iRe[fi]; oIm[o] = iIm[fi];
         fi += fstride;
      }
   }
   else
   {
      for (o = oo; o < oend; o += m)
      {
         this.work(oRe, oIm, o, iRe, iIm, fi, fstride * p, fidx + 2);
         fi += fstride;
      }
   }
   var twRe = this.twRe, twIm = this.twIm, k, t;
   if (p == 2)
   {
      for (k = 0; k < m; ++k)
      {
         var a = oo + k, b = a + m;
         t = k * fstride;
         var wr = twRe[t], wi = twIm[t];
         var tr = oRe[b] * wr - oIm[b] * wi, ti = oRe[b] * wi + oIm[b] * wr;
         oRe[b] = oRe[a] - tr; oIm[b] = oIm[a] - ti;
         oRe[a] += tr; oIm[a] += ti;
      }
   }
   else if (p == 4)
   {
      var m2 = 2 * m, m3 = 3 * m;
      for (k = 0; k < m; ++k)
      {
         var i0 = oo + k, i1 = i0 + m, i2 = i0 + m2, i3 = i0 + m3;
         var t1 = k * fstride, t2 = 2 * t1, t3 = 3 * t1;
         var s0r = oRe[i1] * twRe[t1] - oIm[i1] * twIm[t1], s0i = oRe[i1] * twIm[t1] + oIm[i1] * twRe[t1];
         var s1r = oRe[i2] * twRe[t2] - oIm[i2] * twIm[t2], s1i = oRe[i2] * twIm[t2] + oIm[i2] * twRe[t2];
         var s2r = oRe[i3] * twRe[t3] - oIm[i3] * twIm[t3], s2i = oRe[i3] * twIm[t3] + oIm[i3] * twRe[t3];
         var f0r = oRe[i0], f0i = oIm[i0];
         var s5r = f0r - s1r, s5i = f0i - s1i;
         f0r += s1r; f0i += s1i;
         var s3r = s0r + s2r, s3i = s0i + s2i;
         var s4r = s0r - s2r, s4i = s0i - s2i;
         oRe[i2] = f0r - s3r; oIm[i2] = f0i - s3i;
         oRe[i0] = f0r + s3r; oIm[i0] = f0i + s3i;
         oRe[i1] = s5r + s4i; oIm[i1] = s5i - s4r;
         oRe[i3] = s5r - s4i; oIm[i3] = s5i + s4r;
      }
   }
   else
   {
      var n = this.n, scRe = this.scRe, scIm = this.scIm, u, q1, q, kk, twidx;
      for (u = 0; u < m; ++u)
      {
         kk = oo + u;
         for (q1 = 0; q1 < p; ++q1) { scRe[q1] = oRe[kk]; scIm[q1] = oIm[kk]; kk += m; }
         kk = oo + u;
         var kidx = u;                     // logical output index inside this block
         for (q1 = 0; q1 < p; ++q1)
         {
            twidx = 0;
            var accR = scRe[0], accI = scIm[0];
            for (q = 1; q < p; ++q)
            {
               twidx += fstride * kidx;
               if (twidx >= n) twidx -= n;
               accR += scRe[q] * twRe[twidx] - scIm[q] * twIm[twidx];
               accI += scRe[q] * twIm[twidx] + scIm[q] * twRe[twidx];
            }
            oRe[kk] = accR; oIm[kk] = accI;
            kk += m; kidx += m;
         }
      }
   }
};

var FD_planCache = {};
function FD_plan(n)
{
   if (!FD_planCache[n]) FD_planCache[n] = new FD_FFTPlan(n);
   return FD_planCache[n];
}

// ---------------------------------------------------------------------
// Real 2-D FFT to half spectrum.  Storage: index = kx*H + ky,
// kx in [0, W/2], ky in [0, H).  Two real rows packed per complex FFT.
// ---------------------------------------------------------------------
function FD_forward2D(data, W, H, progress)
{
   var Wh = Math.floor(W / 2) + 1;
   var Sre = new Float64Array(Wh * H), Sim = new Float64Array(Wh * H);
   var pW = FD_plan(W), pH = FD_plan(H);
   var n = Math.max(W, H);
   var bRe = new Float64Array(n), bIm = new Float64Array(n);
   var oRe = new Float64Array(n), oIm = new Float64Array(n);
   var x, y, k;
   for (y = 0; y < H; y += 2)
   {
      var two = (y + 1 < H), r1 = y * W, r2 = r1 + W;
      for (x = 0; x < W; ++x)
      {
         bRe[x] = data[r1 + x];
         bIm[x] = two ? data[r2 + x] : 0;
      }
      pW.transform(bRe, bIm, oRe, oIm);
      for (k = 0; k < Wh; ++k)
      {
         var nk = (k == 0) ? 0 : W - k;
         var zr = oRe[k], zi = oIm[k], cr = oRe[nk], ci = -oIm[nk];
         var idx = k * H + y;
         Sre[idx] = 0.5 * (zr + cr); Sim[idx] = 0.5 * (zi + ci);
         if (two)
         {
            Sre[idx + 1] = 0.5 * (zi - ci);
            Sim[idx + 1] = -0.5 * (zr - cr);
         }
      }
      if (progress && (y & 511) == 0) progress("FFT rows", y / H);
   }
   for (k = 0; k < Wh; ++k)
   {
      var base = k * H;
      for (y = 0; y < H; ++y) { bRe[y] = Sre[base + y]; bIm[y] = Sim[base + y]; }
      pH.transform(bRe, bIm, oRe, oIm);
      for (y = 0; y < H; ++y) { Sre[base + y] = oRe[y]; Sim[base + y] = oIm[y]; }
      if (progress && (k & 511) == 0) progress("FFT columns", k / Wh);
   }
   return { re: Sre, im: Sim, Wh: Wh };
}

// Inverse of FD_forward2D (destroys S). Returns Float32Array W*H.
function FD_inverse2D(S, W, H, progress)
{
   var Wh = S.Wh, Sre = S.re, Sim = S.im;
   var pW = FD_plan(W), pH = FD_plan(H);
   var n = Math.max(W, H);
   var bRe = new Float64Array(n), bIm = new Float64Array(n);
   var oRe = new Float64Array(n), oIm = new Float64Array(n);
   var x, y, k;
   for (k = 0; k < Wh; ++k)
   {
      var base = k * H;
      for (y = 0; y < H; ++y) { bRe[y] = Sre[base + y]; bIm[y] = -Sim[base + y]; }
      pH.transform(bRe, bIm, oRe, oIm);
      for (y = 0; y < H; ++y) { Sre[base + y] = oRe[y]; Sim[base + y] = -oIm[y]; }
      if (progress && (k & 511) == 0) progress("IFFT columns", k / Wh);
   }
   var out = new Float32Array(W * H), norm = 1 / (W * H);
   for (y = 0; y < H; y += 2)
   {
      var two = (y + 1 < H);
      for (k = 0; k < Wh; ++k)
      {
         var i1 = k * H + y;
         var x1r = Sre[i1], x1i = Sim[i1];
         var x2r = two ? Sre[i1 + 1] : 0, x2i = two ? Sim[i1 + 1] : 0;
         // Z = X1 + i X2, conjugated for inverse-via-forward
         bRe[k] = x1r - x2i; bIm[k] = -(x1i + x2r);
      }
      for (k = Wh; k < W; ++k)
      {
         var j = W - k, i2 = j * H + y;
         var a1r = Sre[i2], a1i = Sim[i2];
         var a2r = two ? Sre[i2 + 1] : 0, a2i = two ? Sim[i2 + 1] : 0;
         bRe[k] = a1r + a2i; bIm[k] = -(-a1i + a2r);
      }
      pW.transform(bRe, bIm, oRe, oIm);
      var r1 = y * W, r2 = r1 + W;
      for (x = 0; x < W; ++x)
      {
         out[r1 + x] = oRe[x] * norm;
         if (two) out[r2 + x] = -oIm[x] * norm;
      }
      if (progress && (y & 511) == 0) progress("IFFT rows", y / H);
   }
   return out;
}

// ---------------------------------------------------------------------
// Moisan periodic + smooth decomposition, done in the Fourier domain:
// subtracts S_hat (smooth part) from U_hat so the edge-discontinuity
// cross disappears from the spectrum axes (exactly where the banding
// notches act).  Reference: L. Moisan, "Periodic plus smooth image
// decomposition", J. Math. Imaging Vis. 39 (2011) 161-179.
// ---------------------------------------------------------------------
function FD_periodicPart(S, data, W, H)
{
   var Wh = S.Wh, x, y, q, r;
   var aRe = new Float64Array(H), aIm = new Float64Array(H);
   var ARe = new Float64Array(H), AIm = new Float64Array(H);
   for (y = 0; y < H; ++y) aRe[y] = data[y * W + W - 1] - data[y * W];
   FD_plan(H).transform(aRe, aIm, ARe, AIm);
   var bRe = new Float64Array(W), bIm = new Float64Array(W);
   var BRe = new Float64Array(W), BIm = new Float64Array(W);
   for (x = 0; x < W; ++x) bRe[x] = data[(H - 1) * W + x] - data[x];
   FD_plan(W).transform(bRe, bIm, BRe, BIm);
   var cyR = new Float64Array(H), cyI = new Float64Array(H), cosy = new Float64Array(H);
   for (r = 0; r < H; ++r)
   {
      var ay = 2 * Math.PI * r / H;
      cyR[r] = 1 - Math.cos(ay); cyI[r] = -Math.sin(ay);   // 1 - e^{i ay}
      cosy[r] = 2 * Math.cos(ay);
   }
   for (q = 0; q < Wh; ++q)
   {
      var ax = 2 * Math.PI * q / W;
      var cxR = 1 - Math.cos(ax), cxI = -Math.sin(ax);
      var cosx = 2 * Math.cos(ax), bqR = BRe[q], bqI = BIm[q], base = q * H;
      for (r = 0; r < H; ++r)
      {
         if (q == 0 && r == 0) continue;
         var vr = ARe[r] * cxR - AIm[r] * cxI + bqR * cyR[r] - bqI * cyI[r];
         var vi = ARe[r] * cxI + AIm[r] * cxR + bqR * cyI[r] + bqI * cyR[r];
         var den = cosx + cosy[r] - 4;
         S.re[base + r] -= vr / den;
         S.im[base + r] -= vi / den;
      }
   }
}

// ---------------------------------------------------------------------
// Robust helpers
// ---------------------------------------------------------------------
function FD_select(a, n, k)
{
   // quickselect on first n entries of a (modified)
   var l = 0, r = n - 1;
   while (r > l)
   {
      var pv = a[(l + r) >> 1], i = l, j = r;
      do
      {
         while (a[i] < pv) ++i;
         while (pv < a[j]) --j;
         if (i <= j) { var t = a[i]; a[i] = a[j]; a[j] = t; ++i; --j; }
      }
      while (i <= j);
      if (j < k) l = i;
      if (k < i) r = j;
   }
   return a[k];
}
function FD_median(a, n) { return FD_select(a, n, n >> 1); }

// Large-scale background by block medians + bilinear interpolation.
// get(ix, iy) returns sample; returns {bg: function(ix,iy), grid...}
function FD_blockBackground(get, nx, ny, B, step)
{
   var gw = Math.ceil(nx / B), gh = Math.ceil(ny / B);
   var grid = new Float64Array(gw * gh), buf = new Float64Array(B * B), i, j;
   for (j = 0; j < gh; ++j)
      for (i = 0; i < gw; ++i)
      {
         var n = 0, x0 = i * B, y0 = j * B;
         var x1 = Math.min(nx, x0 + B), y1 = Math.min(ny, y0 + B);
         for (var y = y0; y < y1; y += step)
            for (var x = x0; x < x1; x += step)
            {
               var gv = get(x, y);
               if (gv === gv) buf[n++] = gv;           // NaN = masked sample
            }
         grid[j * gw + i] = n >= 8 ? FD_median(buf, n) : NaN;
      }
   // blocks with too few unmasked samples: fill from valid neighbours (outward)
   for (var pass = 0; pass < gw + gh; ++pass)
   {
      var holes = 0, nxt = new Float64Array(grid);      // copy (TypedArray.slice is ES2015)
      for (j = 0; j < gh; ++j)
         for (i = 0; i < gw; ++i)
         {
            if (grid[j * gw + i] === grid[j * gw + i]) continue;
            var sm = 0, cnt = 0;
            for (var dj = -1; dj <= 1; ++dj)
               for (var di = -1; di <= 1; ++di)
               {
                  var ii = i + di, jj = j + dj;
                  if (ii < 0 || jj < 0 || ii >= gw || jj >= gh) continue;
                  var q = grid[jj * gw + ii];
                  if (q === q) { sm += q; ++cnt; }
               }
            if (cnt > 0) nxt[j * gw + i] = sm / cnt; else ++holes;
         }
      grid = nxt;
      if (holes == 0) break;
   }
   for (i = 0; i < grid.length; ++i) if (grid[i] !== grid[i]) grid[i] = 0;
   var f = function(x, y)
   {
      var fx = (x + 0.5) / B - 0.5, fy = (y + 0.5) / B - 0.5;
      if (fx < 0) fx = 0; if (fx > gw - 1) fx = gw - 1;
      if (fy < 0) fy = 0; if (fy > gh - 1) fy = gh - 1;
      var ix = Math.floor(fx), iy = Math.floor(fy);
      var ix1 = Math.min(ix + 1, gw - 1), iy1 = Math.min(iy + 1, gh - 1);
      var tx = fx - ix, ty = fy - iy;
      var a = grid[iy * gw + ix], b = grid[iy * gw + ix1];
      var c = grid[iy1 * gw + ix], d = grid[iy1 * gw + ix1];
      return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
   };
   // Fast dense evaluation. rowOf(y) maps output row y to a grid-space row;
   // dstIndex(x, y) gives the destination index.
   f.fill = function(dst, rowOf, dstIndex)
   {
      var ix0 = new Int32Array(nx), ix1 = new Int32Array(nx), txs = new Float64Array(nx), line = new Float64Array(gw), x, y, k;
      for (x = 0; x < nx; ++x)
      {
         var fx = (x + 0.5) / B - 0.5;
         if (fx < 0) fx = 0; if (fx > gw - 1) fx = gw - 1;
         ix0[x] = Math.floor(fx); ix1[x] = Math.min(ix0[x] + 1, gw - 1); txs[x] = fx - ix0[x];
      }
      for (y = 0; y < ny; ++y)
      {
         var gy = rowOf(y), fy = (gy + 0.5) / B - 0.5;
         if (fy < 0) fy = 0; if (fy > gh - 1) fy = gh - 1;
         var iy = Math.floor(fy), iy1 = Math.min(iy + 1, gh - 1), ty = fy - iy;
         for (k = 0; k < gw; ++k) line[k] = grid[iy * gw + k] * (1 - ty) + grid[iy1 * gw + k] * ty;
         for (x = 0; x < nx; ++x)
            dst[dstIndex(x, y)] = line[ix0[x]] * (1 - txs[x]) + line[ix1[x]] * txs[x];
      }
   };
   return f;
}

// ---------------------------------------------------------------------
// Separable helpers for the protection mask
// ---------------------------------------------------------------------
function FD_boxBlur(src, dst, W, H, r)
{
   // (2r+1)^2 box mean, edge-clamped, two running-sum passes; tmp row/col buffers
   var x, y, k, n = 2 * r + 1, line = new Float64Array(Math.max(W, H) + 2 * r);
   var tmp = new Float32Array(W * H);
   for (y = 0; y < H; ++y)
   {
      var o = y * W;
      for (k = 0; k < W + 2 * r; ++k) { var xx = k - r; xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx); line[k] = src[o + xx]; }
      var acc = 0;
      for (k = 0; k < n; ++k) acc += line[k];
      for (x = 0; x < W; ++x) { tmp[o + x] = acc / n; acc += line[x + n] - line[x]; }
   }
   for (x = 0; x < W; ++x)
   {
      for (k = 0; k < H + 2 * r; ++k) { var yy = k - r; yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy); line[k] = tmp[yy * W + x]; }
      var acc2 = 0;
      for (k = 0; k < n; ++k) acc2 += line[k];
      for (y = 0; y < H; ++y) { dst[y * W + x] = acc2 / n; acc2 += line[y + n] - line[y]; }
   }
}

function FD_dilate(mask, W, H, r)
{
   // square dilation by r px (separable, run-length counting)
   if (r < 1) return;
   var x, y, cnt, tmp = new Uint8Array(W * H);
   for (y = 0; y < H; ++y)
   {
      var o = y * W; cnt = 0;
      for (x = 0; x < Math.min(r, W); ++x) cnt += mask[o + x];
      for (x = 0; x < W; ++x)
      {
         if (x + r < W) cnt += mask[o + x + r];
         if (x - r - 1 >= 0) cnt -= mask[o + x - r - 1];
         tmp[o + x] = cnt > 0 ? 1 : 0;
      }
   }
   for (x = 0; x < W; ++x)
   {
      cnt = 0;
      for (y = 0; y < Math.min(r, H); ++y) cnt += tmp[y * W + x];
      for (y = 0; y < H; ++y)
      {
         if (y + r < H) cnt += tmp[(y + r) * W + x];
         if (y - r - 1 >= 0) cnt -= tmp[(y - r - 1) * W + x];
         mask[y * W + x] = cnt > 0 ? 1 : 0;
      }
   }
}

// Subtract the HIGH-PASSED part of robust row and column medians (over unmasked
// pixels).  Only the band-like component is removed: a smooth bias (halo tail,
// gradient) in the medians must not be pushed back into whole rows/columns.
function FD_hpOffsets(med, n, half)
{
   var out = new Float64Array(n), buf = new Float64Array(2 * half + 1), i, k, m;
   for (i = 0; i < n; ++i)
   {
      if (med[i] !== med[i]) { out[i] = 0; continue; }
      m = 0;
      for (k = Math.max(0, i - half); k <= Math.min(n - 1, i + half); ++k) if (med[k] === med[k]) buf[m++] = med[k];
      out[i] = med[i] - FD_median(buf, m);
   }
   return out;
}

function FD_removeRowColMedians(a, mask, W, H)
{
   var x, y, n, buf = new Float64Array(Math.max(W, H)), sx = Math.max(1, Math.floor(W / 2000)), sy = Math.max(1, Math.floor(H / 2000));
   // No minimum unmasked fraction: only the high-passed (band-like) part of the
   // medians is removed, so a smooth bias from masked halos cannot leak in, and a
   // heavily masked row MUST still have its band removed, otherwise the band stays
   // in the residual and the hysteresis mask grows along the whole row (1.3 bug).
   var rm = new Float64Array(H), cm = new Float64Array(W), need = 16;
   for (y = 0; y < H; ++y)
   {
      var o = y * W; n = 0;
      for (x = 0; x < W; x += sx) if (!mask || !mask[o + x]) buf[n++] = a[o + x];
      rm[y] = n >= need ? FD_median(buf, n) : NaN;
   }
   rm = FD_hpOffsets(rm, H, 32);
   for (y = 0; y < H; ++y) { var o2 = y * W, r = rm[y]; if (r) for (x = 0; x < W; ++x) a[o2 + x] -= r; }
   for (x = 0; x < W; ++x)
   {
      n = 0;
      for (y = 0; y < H; y += sy) if (!mask || !mask[y * W + x]) buf[n++] = a[y * W + x];
      cm[x] = n >= need ? FD_median(buf, n) : NaN;
   }
   cm = FD_hpOffsets(cm, W, 32);
   for (x = 0; x < W; ++x) { var c = cm[x]; if (c) for (y = 0; y < H; ++y) a[y * W + x] -= c; }
}

// Per-pixel noise sigma from horizontal first differences (insensitive to smooth structure and row bands).
function FD_diffSigma(d, W, H)
{
   var n = Math.min(400000, (W - 1) * H), st = Math.max(1, Math.floor((W - 1) * H / n)), buf = new Float64Array(n + 1), k = 0, i;
   for (i = 0; i < W * H - 1 && k < n; i += st) if ((i + 1) % W) buf[k++] = d[i + 1] - d[i];
   var m = FD_median(buf, k);
   for (i = 0; i < k; ++i) buf[i] = Math.abs(buf[i] - m);
   return 1.4826 * FD_median(buf, k) / Math.SQRT2;
}

function FD_robustSigma(a, mask, N)
{
   var ns = Math.min(N, 400000), stride = Math.max(1, Math.floor(N / ns));
   var rs = new Float64Array(Math.ceil(N / stride) + 1), nr = 0, i;
   for (i = 0; i < N; i += stride) if (!mask || !mask[i]) rs[nr++] = a[i];
   if (nr < 100) for (nr = 0, i = 0; i < N; i += stride) rs[nr++] = a[i];
   var med = FD_median(rs, nr);
   for (i = 0; i < nr; ++i) rs[i] = Math.abs(rs[i] - med);
   return { med: med, sigma: 1.4826 * FD_median(rs, nr) };
}

// ---------------------------------------------------------------------
// Protection mask.  Stars, their halos, diffraction spikes, bleed columns,
// compact bright nebulae and dark defects: anything localized that would
// otherwise leak into the row/column means the band notches estimate.
//  1. background: 64-px block medians of UNMASKED pixels, iterated 3x, so
//     a bright star (saturated or not) cannot drag it up with it;
//  2. seeds: 5x5-smoothed |residual| > detectK * sigma_s, plus saturated pixels;
//  3. hysteresis growth over smoothed |residual| > maskLow * sigma_s
//     (follows a halo down to the noise floor);
//  4. square dilation by maskGrow px.
// ---------------------------------------------------------------------
function FD_buildMask(data, W, H, P, log)
{
   var N = W * H, i, x, y, it;
   var mask = new Uint8Array(N), bg = new Float32Array(N), res = new Float32Array(N), sm = new Float32Array(N);
   var B = 64, st, sigS = 0;
   // saturation level: the data maximum, if a meaningful number of pixels sit there
   var vmax = 0, nsat = 0;
   for (i = 0; i < N; ++i) if (data[i] > vmax) vmax = data[i];
   var satLevel = vmax * (1 - 1e-4);
   for (i = 0; i < N; ++i) if (data[i] >= satLevel) ++nsat;
   var useSat = nsat >= 9 && vmax > 0;
   if (useSat) log("  " + nsat + " saturated pixels (>= " + satLevel.toPrecision(5) + ") always protected");
   for (it = 0; it < 3; ++it)
   {
      var bgf = FD_blockBackground(function(x, y) { var j = y * W + x; return mask[j] ? NaN : data[j]; }, W, H, B, 4);
      bgf.fill(bg, function(y) { return y; }, function(x, y) { return y * W + x; });
      for (i = 0; i < N; ++i) res[i] = data[i] - bg[i];
      FD_removeRowColMedians(res, it > 0 ? mask : null, W, H);   // banding must not be "structure"
      FD_boxBlur(res, sm, W, H, 2);
      st = FD_robustSigma(sm, it > 0 ? mask : null, N);
      sigS = st.sigma;
      var hi = P.detectK * sigS, lo = P.maskLow * sigS, m0 = st.med;
      // seeds
      for (i = 0; i < N; ++i) { var d = sm[i] - m0; mask[i] = (d > hi || d < -hi || (useSat && data[i] >= satLevel)) ? 1 : 0; }
      // hysteresis: flood fill from the seeds through 4-connected pixels whose
      // smoothed |residual| exceeds the halo threshold (cost ~ masked pixels)
      var cap = 1 << 20, stack = new Int32Array(cap), sp = 0;
      for (i = 0; i < N; ++i)
         if (mask[i])
         {
            if (sp == cap) { var ns2 = new Int32Array(cap *= 2); ns2.set(stack); stack = ns2; }
            stack[sp++] = i;
         }
      while (sp > 0)
      {
         var j0 = stack[--sp], x0 = j0 % W;
         for (var nbk = 0; nbk < 4; ++nbk)
         {
            var j;
            if (nbk == 0) { if (x0 == 0) continue; j = j0 - 1; }
            else if (nbk == 1) { if (x0 == W - 1) continue; j = j0 + 1; }
            else if (nbk == 2) { if (j0 < W) continue; j = j0 - W; }
            else { j = j0 + W; if (j >= N) continue; }
            if (mask[j]) continue;
            var dd = sm[j] - m0;
            if (dd <= lo && dd >= -lo && !(useSat && data[j] >= satLevel)) continue;
            mask[j] = 1;
            if (sp == cap) { var ns3 = new Int32Array(cap *= 2); ns3.set(stack); stack = ns3; }
            stack[sp++] = j;
         }
      }
      stack = null;
      FD_dilate(mask, W, H, Math.round(P.maskGrow));
   }
   // smooth the background used as FILL: bilinear block interpolation has kinks
   // along every block row/column, which the band notches would read as bands
   var tmpb = new Float32Array(N);
   FD_boxBlur(bg, tmpb, W, H, B >> 1);
   FD_boxBlur(tmpb, bg, W, H, B >> 1);
   tmpb = null;
   var sig = FD_robustSigma(res, mask, N);
   var cnt = 0;
   for (i = 0; i < N; ++i) cnt += mask[i];
   res = null; sm = null;
   log("  noise sigma = " + sig.sigma.toExponential(3) + ", protection mask covers " + (100 * cnt / N).toFixed(2) + "% of pixels");
   return { mask: mask, bg: bg, sigma: sig.sigma, frac: cnt / N };
}

// ---------------------------------------------------------------------
// Fourier noise model of one estimation image (steps 2-5).
// Returns { noise, diag, spikes }
// ---------------------------------------------------------------------
function FD_noiseModel(est, W, H, P, progress, wantDiag, wantComps)
{
   var N = W * H, i, x, y;
   var log = progress ? function(s) { progress(s, -1); } : function() {};
   // ---- 2. Spectrum of the periodic component
   var S = FD_forward2D(est, W, H, progress);
   FD_periodicPart(S, est, W, H);
   var Wh = S.Wh, M = Wh * H, Sre = S.re, Sim = S.im;
   var halfH = Math.floor(H / 2);

   // frequency helpers
   var f0 = 1 / Math.max(2, P.protectScale);
   function ramp(f) { f = Math.abs(f); if (f <= f0) return 0; if (f >= 2 * f0) return 1; return 0.5 - 0.5 * Math.cos(Math.PI * (f - f0) / f0); }
   var rampY = new Float32Array(H), gaussY = new Float32Array(H), nyqY = new Float32Array(H), kyc = new Int32Array(H);
   var sl = P.lineSigma, sn = P.nyquistSigma;
   for (y = 0; y < H; ++y)
   {
      var c = (y <= halfH) ? y : y - H;
      kyc[y] = c;
      rampY[y] = ramp(c / H);
      gaussY[y] = Math.exp(-c * c / (2 * sl * sl));
      var d = H / 2 - Math.abs(c);
      nyqY[y] = Math.exp(-d * d / (2 * sn * sn));
   }

   // ---- 3. Periodic spike detection (log magnitude vs local robust bg)
   var spikeGain = null, nSpikes = 0;
   var L = null;
   if (P.spikes || wantDiag)
   {
      log("Log-magnitude spectrum");
      L = new Float32Array(M);
      for (i = 0; i < M; ++i) L[i] = 0.5 * Math.log(Sre[i] * Sre[i] + Sim[i] * Sim[i] + 1e-30);
   }
   var excl = 2 * f0;
   function excluded(kx, ky)
   {
      var c = kyc[ky];
      if (kx <= 1 || c <= 1 && c >= -1) return true;         // axes (banding territory)
      var fx = kx / W, fy = c / H;
      return fx * fx + fy * fy < excl * excl;
   }
   if (P.spikes)
   {
      log("Periodic spike detection");
      // centered-ky accessor for block background
      var Lget = function(kx, cy) { var ky = (cy - halfH + H) % H; return L[kx * H + ky]; };
      var bgL = FD_blockBackground(Lget, Wh, H, 16, 1);
      // robust sigma of (L - bg) outside exclusions
      var smp = new Float64Array(300000), ns2 = 0;
      var st = Math.max(1, Math.floor(M / 300000));
      for (i = 0; i < M && ns2 < 300000; i += st)
      {
         var kx = Math.floor(i / H), ky = i - kx * H;
         if (excluded(kx, ky)) continue;
         smp[ns2++] = L[i] - bgL(kx, (ky + halfH) % H);
      }
      var m0 = FD_median(smp, ns2);
      for (i = 0; i < ns2; ++i) smp[i] = Math.abs(smp[i] - m0);
      var sL = 1.4826 * FD_median(smp, ns2);
      log("  log-spectrum robust sigma = " + sL.toFixed(3));
      spikeGain = new Float32Array(M);
      for (i = 0; i < M; ++i) spikeGain[i] = 1;
      var bgArr = new Float32Array(M);
      // grid axes are (kx, centered cy); here the output "row" is ky
      bgL.fill(bgArr, function(ky) { return (ky + halfH) % H; }, function(kx, ky) { return kx * H + ky; });
      for (i = 0; i < M; ++i) bgArr[i] += m0;
      var thr = P.spikeK * sL, dl = Math.round(P.spikeDilate), hits = [];
      for (kx = 0; kx < Wh; ++kx)
         for (ky = 0; ky < H; ++ky)
         {
            i = kx * H + ky;
            if (excluded(kx, ky)) continue;
            if (L[i] - bgArr[i] > thr) hits.push(i);
         }
      nSpikes = hits.length;
      for (var h = 0; h < hits.length; ++h)
      {
         var hx = Math.floor(hits[h] / H), hy = hits[h] - hx * H;
         for (var dy = -dl; dy <= dl; ++dy)
            for (var dx = -dl; dx <= dl; ++dx)
            {
               var qx = hx + dx, qy = hy + dy;
               if (qx < 0) { qx = -qx; qy = -qy; }            // Hermitian mirror
               if (qx >= Wh) continue;
               qy = ((qy % H) + H) % H;
               if (excluded(qx, qy)) continue;
               var j = qx * H + qy, e = L[j] - bgArr[j];
               if (e > 0)
               {
                  var g = Math.exp(-e);
                  if (g < spikeGain[j]) spikeGain[j] = g;
               }
            }
      }
      bgArr = null;
      // Hermitian consistency on self-conjugate columns (kx=0, kx=W/2)
      var selfCols = [0];
      if (W % 2 == 0) selfCols.push(Wh - 1);
      for (var sc = 0; sc < selfCols.length; ++sc)
      {
         var b0 = selfCols[sc] * H;
         for (ky = 1; ky < H; ++ky)
         {
            var jj = H - ky, g2 = Math.min(spikeGain[b0 + ky], spikeGain[b0 + jj]);
            spikeGain[b0 + ky] = spikeGain[b0 + jj] = g2;
         }
      }
      log("  " + nSpikes + " spectral spike bins above " + P.spikeK + " sigma");
   }

   // ---- 4. Build total gain, form noise spectrum = (1-G) * P_hat
   var s = P.strength;
   var gainOut = wantDiag ? new Float32Array(M) : null;
   // for adaptive amplitude matching: keep the periodic spectrum (float32) to synthesize the
   // row-band and column-band components separately after the total model
   var keepRe = wantComps ? new Float32Array(M) : null, keepIm = wantComps ? new Float32Array(M) : null;
   if (wantComps) for (i = 0; i < M; ++i) { keepRe[i] = Sre[i]; keepIm[i] = Sim[i]; }
   for (kx = 0; kx < Wh; ++kx)
   {
      var gx = Math.exp(-kx * kx / (2 * sl * sl));
      var rx = ramp(kx / W);
      var dnx = W / 2 - kx, nyx = Math.exp(-dnx * dnx / (2 * sn * sn));
      var base = kx * H;
      for (ky = 0; ky < H; ++ky)
      {
         var G = 1;
         if (P.horizontal) G *= 1 - s * gx * rampY[ky];      // row offsets -> fx ~ 0
         if (P.vertical)   G *= 1 - s * gaussY[ky] * rx;     // column offsets -> fy ~ 0
         if (P.nyquist)    G *= 1 - s * (1 - (1 - nyx) * (1 - nyqY[ky]));
         if (spikeGain)    G *= 1 - s * (1 - spikeGain[base + ky]);
         if (kx == 0 && ky == 0) G = 1;                       // never touch DC
         var idx = base + ky, rem = 1 - G;
         Sre[idx] *= rem; Sim[idx] *= rem;
         if (gainOut) gainOut[idx] = G;
      }
   }
   spikeGain = null;

   // ---- 5. Diagnostic spectrum image (centered, log |P|, removed part in red)
   var diag = null;
   if (wantDiag)
   {
      log("Building spectrum diagnostic");
      var smpD = new Float64Array(200000), nd = 0, sd = Math.max(1, Math.floor(M / 200000)), Lmax = -1e30;
      for (i = 0; i < M; ++i) if (L[i] > Lmax && i != 0) Lmax = L[i];
      for (i = 1; i < M && nd < 200000; i += sd) smpD[nd++] = L[i];
      var Llo = FD_median(smpD, nd) - 1.0;
      var span = Math.max(1e-6, Lmax - Llo);
      var dR = new Float32Array(N), dG = new Float32Array(N);
      var cx = Math.floor(W / 2), cyy = Math.floor(H / 2);
      for (y = 0; y < H; ++y)
         for (x = 0; x < W; ++x)
         {
            var kxs = x - cx, kys = y - cyy;
            if (kxs < 0) { kxs = -kxs; kys = -kys; }
            if (kxs >= Wh) kxs = Wh - 1;
            var idd = kxs * H + ((kys % H) + H) % H;
            var v = (L[idd] - Llo) / span; if (v < 0) v = 0; if (v > 1) v = 1;
            var o = y * W + x;
            dR[o] = v; dG[o] = v * gainOut[idd];
         }
      diag = { r: dR, g: dG };
      gainOut = null;
   }
   L = null;

   var noise = FD_inverse2D(S, W, H, progress);
   var comps = null;
   if (wantComps)
   {
      // n_h = F^-1{ s A_h p^ },  n_v = F^-1{ s A_v p^ }; the rest (spikes, Nyquist, cross terms) is
      // n - n_h - n_v and is subtracted with unit gain
      comps = {};
      var which = [];
      if (P.horizontal) which.push("h");
      if (P.vertical) which.push("v");
      for (var c2 = 0; c2 < which.length; ++c2)
      {
         log("Band component " + (which[c2] == "h" ? "rows" : "columns"));
         for (kx = 0; kx < Wh; ++kx)
         {
            var gx2 = Math.exp(-kx * kx / (2 * sl * sl)), rx2 = ramp(kx / W), base2 = kx * H;
            for (ky = 0; ky < H; ++ky)
            {
               var A = (which[c2] == "h") ? s * gx2 * rampY[ky] : s * gaussY[ky] * rx2;
               if (kx == 0 && ky == 0) A = 0;
               Sre[base2 + ky] = keepRe[base2 + ky] * A;
               Sim[base2 + ky] = keepIm[base2 + ky] * A;
            }
         }
         comps[which[c2]] = FD_inverse2D(S, W, H, progress);
      }
      keepRe = keepIm = null;
   }
   S = null;
   return { noise: noise, diag: diag, spikes: nSpikes, comps: comps };
}

// ---------------------------------------------------------------------
// Adaptive amplitude matching.  For each band component c (rows, columns) a smoothly
// varying gain a_c(x,y) is fitted by windowed least squares over the UNMASKED pixels:
//    y = d - b~  ~  offset + sum_c a_c n_c        (offset = nuisance, never subtracted)
// with a Gaussian prior a_c ~ N(1, tau^2) (posterior mean): a window with little band
// energy keeps a ~ 1, a window with strong bands follows the data.  Gains are clamped
// to [0, 2] and smoothed.  Computed on a grid of Bk x Bk blocks (memory ~ N / Bk^2).
// Returns { h: Float32Array|null, v: Float32Array|null, stats }  (per-pixel gains)
// ---------------------------------------------------------------------
function FD_adaptiveGains(d, bg, mask, comps, W, H, halfWin, sigma, tau, log)
{
   var Bk = 8, gw = Math.ceil(W / Bk), gh = Math.ceil(H / Bk), G2 = gw * gh, x, y, i, j;
   var nh = comps.h || null, nv = comps.v || null;
   // per-block sums: w, h, v, y, hh, vv, hv, hy, vy
   var K = 9, sums = [];
   for (i = 0; i < K; ++i) sums.push(new Float64Array(G2));
   for (y = 0; y < H; ++y)
   {
      var by = (y / Bk) | 0, o = y * W;
      for (x = 0; x < W; ++x)
      {
         var p = o + x;
         if (mask && mask[p]) continue;
         var b = by * gw + ((x / Bk) | 0), yy = d[p] - (bg ? bg[p] : 0), hh = nh ? nh[p] : 0, vv = nv ? nv[p] : 0;
         sums[0][b] += 1; sums[1][b] += hh; sums[2][b] += vv; sums[3][b] += yy;
         sums[4][b] += hh * hh; sums[5][b] += vv * vv; sums[6][b] += hh * vv; sums[7][b] += hh * yy; sums[8][b] += vv * yy;
      }
   }
   // window sums over (2 rb + 1)^2 blocks: separable running sums
   var rb = Math.max(1, Math.round(halfWin / Bk));
   function boxGrid(a)
   {
      var t = new Float64Array(G2), out = new Float64Array(G2), xx, yy2, acc;
      for (yy2 = 0; yy2 < gh; ++yy2)
      {
         acc = 0;
         for (xx = -rb; xx < gw + rb; ++xx)
         {
            if (xx + rb < gw && xx + rb >= 0) acc += a[yy2 * gw + xx + rb];
            if (xx - rb - 1 >= 0 && xx - rb - 1 < gw) acc -= a[yy2 * gw + xx - rb - 1];
            if (xx >= 0 && xx < gw) t[yy2 * gw + xx] = acc;
         }
      }
      for (xx = 0; xx < gw; ++xx)
      {
         acc = 0;
         for (yy2 = -rb; yy2 < gh + rb; ++yy2)
         {
            if (yy2 + rb < gh && yy2 + rb >= 0) acc += t[(yy2 + rb) * gw + xx];
            if (yy2 - rb - 1 >= 0 && yy2 - rb - 1 < gh) acc -= t[(yy2 - rb - 1) * gw + xx];
            if (yy2 >= 0 && yy2 < gh) out[yy2 * gw + xx] = acc;
         }
      }
      return out;
   }
   for (i = 0; i < K; ++i) sums[i] = boxGrid(sums[i]);
   var full = (2 * rb + 1) * (2 * rb + 1) * Bk * Bk, s2 = sigma * sigma, t2 = tau * tau;
   var ah = new Float64Array(G2), av = new Float64Array(G2);
   for (j = 0; j < G2; ++j)
   {
      var n = sums[0][j];
      ah[j] = av[j] = 1;
      if (n < 0.2 * full) continue;                     // window mostly masked: keep the model as is
      var mh = sums[1][j] / n, mv = sums[2][j] / n, my = sums[3][j] / n;
      var chh = sums[4][j] / n - mh * mh, cvv = sums[5][j] / n - mv * mv, chv = sums[6][j] / n - mh * mv;
      var chy = sums[7][j] / n - mh * my, cvy = sums[8][j] / n - mv * my;
      var q = s2 / (n * t2);                            // prior precision relative to the data term
      if (nh && nv)
      {
         var A11 = chh + q, A22 = cvv + q, det = A11 * A22 - chv * chv;
         if (Math.abs(det) > 1e-300) { ah[j] = ((chy + q) * A22 - (cvy + q) * chv) / det; av[j] = (A11 * (cvy + q) - chv * (chy + q)) / det; }
      }
      else if (nh) ah[j] = (chy + q) / (chh + q);
      else av[j] = (cvy + q) / (cvv + q);
      ah[j] = Math.min(2, Math.max(0, ah[j])); av[j] = Math.min(2, Math.max(0, av[j]));
   }
   // smooth the gain fields with the same window (normalized)
   var ones = new Float64Array(G2); for (j = 0; j < G2; ++j) ones[j] = 1;
   var nrm = boxGrid(ones), sh = boxGrid(ah), sv = boxGrid(av);
   for (j = 0; j < G2; ++j) { ah[j] = sh[j] / nrm[j]; av[j] = sv[j] / nrm[j]; }
   // statistics for the console
   function pct(a) { var c = new Float64Array(a); return [FD_select(c, G2, Math.floor(0.05 * (G2 - 1))), FD_select(c, G2, Math.floor(0.5 * (G2 - 1))), FD_select(c, G2, Math.floor(0.95 * (G2 - 1)))]; }
   var stats = { h: nh ? pct(ah) : null, v: nv ? pct(av) : null };
   // bilinear interpolation of the block grid to pixels
   function expand(g)
   {
      var out = new Float32Array(W * H), ix0 = new Int32Array(W), ix1 = new Int32Array(W), tx = new Float64Array(W), xx2;
      for (xx2 = 0; xx2 < W; ++xx2)
      {
         var fx = (xx2 + 0.5) / Bk - 0.5; if (fx < 0) fx = 0; if (fx > gw - 1) fx = gw - 1;
         ix0[xx2] = Math.floor(fx); ix1[xx2] = Math.min(ix0[xx2] + 1, gw - 1); tx[xx2] = fx - ix0[xx2];
      }
      for (var yy3 = 0; yy3 < H; ++yy3)
      {
         var fy = (yy3 + 0.5) / Bk - 0.5; if (fy < 0) fy = 0; if (fy > gh - 1) fy = gh - 1;
         var iy0 = Math.floor(fy), iy1 = Math.min(iy0 + 1, gh - 1), ty = fy - iy0, o3 = yy3 * W;
         for (xx2 = 0; xx2 < W; ++xx2)
         {
            var a0 = g[iy0 * gw + ix0[xx2]] * (1 - tx[xx2]) + g[iy0 * gw + ix1[xx2]] * tx[xx2];
            var a1 = g[iy1 * gw + ix0[xx2]] * (1 - tx[xx2]) + g[iy1 * gw + ix1[xx2]] * tx[xx2];
            out[o3 + xx2] = a0 * (1 - ty) + a1 * ty;
         }
      }
      return out;
   }
   return { h: nh ? expand(ah) : null, v: nv ? expand(av) : null, stats: stats };
}

// ---------------------------------------------------------------------
// Main processing of one channel.
// Returns { out, noise, diag, mask, spikes, rms }
// ---------------------------------------------------------------------
function FD_processChannel(data, W, H, P, progress)
{
   var N = W * H, i;
   var log = progress ? function(s) { progress(s, -1); } : function() {};
   var est = data, M = null;
   if (P.protectStars)
   {
      log("Protection mask");
      M = FD_buildMask(data, W, H, P, log);
      est = new Float32Array(N);
      for (i = 0; i < N; ++i) est[i] = M.mask[i] ? M.bg[i] : data[i];
   }
   // Masked pixels are filled with background + current band model and the
   // model re-estimated (POCS / Gerchberg-Papoulis).  The band estimate then
   // converges to the mean over UNMASKED pixels of each row/column; the bias
   // after n passes scales as (masked fraction of that row)^(n+1).
   var passes = 1 + (M ? Math.round(P.refine) : 0), R = null;
   var adapt = !!P.adaptive && (P.horizontal || P.vertical);
   for (var pass = 0; pass < passes; ++pass)
   {
      var last = (pass == passes - 1);
      if (passes > 1) log("Pass " + (pass + 1) + "/" + passes);
      R = FD_noiseModel(est, W, H, P, progress, last && P.wantDiag, last && adapt);
      if (!last)
         for (i = 0; i < N; ++i) if (M.mask[i]) est[i] = M.bg[i] + R.noise[i];
   }
   est = null;
   var noise = R.noise, gains = null;
   if (adapt)
   {
      log("Adaptive amplitude matching");
      var sig = M ? M.sigma : FD_diffSigma(data, W, H);
      gains = FD_adaptiveGains(data, M ? M.bg : null, M ? M.mask : null, R.comps, W, H, Math.round(P.adaptWindow / 2), sig, 0.1, log);
      var fmt3 = function(t) { return t ? "median " + t[1].toFixed(2) + " (5-95%: " + t[0].toFixed(2) + "-" + t[2].toFixed(2) + ")" : ""; };
      if (gains.h) log("  row-band gain " + fmt3(gains.stats.h));
      if (gains.v) log("  column-band gain " + fmt3(gains.stats.v));
      // n_adapted = n + (a_h - 1) n_h + (a_v - 1) n_v
      for (i = 0; i < N; ++i)
      {
         var dn = 0;
         if (gains.h) dn += (gains.h[i] - 1) * R.comps.h[i];
         if (gains.v) dn += (gains.v[i] - 1) * R.comps.v[i];
         noise[i] += dn;
      }
      R.comps = null;
   }
   var out = new Float32Array(N), ss = 0;
   for (i = 0; i < N; ++i)
   {
      var nv = noise[i], v2 = data[i] - nv;
      ss += nv * nv;
      out[i] = v2 < 0 ? 0 : (v2 > 1 ? 1 : v2);
   }
   return { out: out, noise: P.wantNoise ? noise : null, diag: R.diag,
            mask: (M && P.wantMask) ? M.mask : null, spikes: R.spikes, rms: Math.sqrt(ss / N),
            gains: (gains && P.wantGains) ? gains : null };
}
// ==== CORE END ====

// ==== PJSR USER INTERFACE ====

function FD_defaults()
{
   return {
      horizontal:   true,
      vertical:     true,
      lineSigma:    1.0,
      protectScale: 128,
      spikes:       false,
      spikeK:       6.0,
      spikeDilate:  1,
      nyquist:      false,
      nyquistSigma: 1.5,
      protectStars: true,
      detectK:      5.0,
      maskLow:      2.0,
      maskGrow:     4,
      refine:       1,
      wantMask:     false,
      strength:     1.0,
      inPlace:      false,
      wantNoise:    false,
      wantDiag:     false,
      adaptive:     false,
      adaptWindow:  128,
      wantGains:    false,
      // batch
      batch:        false,
      outputDir:    "",
      postfix:      "_fd",
      overwrite:    false,
      fileList:     ""        // "|"-separated, persisted between sessions
   };
}

var FD_SETTINGS_KEY = "FourierDeband/";

function FD_loadSettings(P)
{
   for (var k in P)
   {
      var t = typeof P[k];
      var dt = (t == "boolean") ? DataType_Boolean : (t == "string") ? DataType_String : DataType_Double;
      try
      {
         var v = Settings.read(FD_SETTINGS_KEY + k, dt);
         if (Settings.lastReadOK && v !== null && v !== undefined) P[k] = v;
      }
      catch (e) {}
   }
}

function FD_saveSettings(P)
{
   for (var k in P)
   {
      var t = typeof P[k];
      var dt = (t == "boolean") ? DataType_Boolean : (t == "string") ? DataType_String : DataType_Double;
      try { Settings.write(FD_SETTINGS_KEY + k, dt, P[k]); } catch (e) {}
   }
}

function FD_readChannel(image, c)
{
   var W = image.width, H = image.height, r = new Rect(0, 0, W, H);
   var a = new Float32Array(W * H);
   try { image.getSamples(a, r, c); return a; }
   catch (e)
   {
      var t = [];
      image.getSamples(t, r, c);
      for (var i = 0; i < t.length; ++i) a[i] = t[i];
      return a;
   }
}

function FD_writeChannel(image, a, c)
{
   var r = new Rect(0, 0, image.width, image.height);
   try { image.setSamples(a, r, c); }
   catch (e)
   {
      var t = new Array(a.length);
      for (var i = 0; i < a.length; ++i) t[i] = a[i];
      image.setSamples(t, r, c);
   }
}

function FD_uniqueId(base)
{
   var id = base.replace(/[^A-Za-z0-9_]/g, "_"), n = 1, cand = id;
   if (!/^[A-Za-z_]/.test(cand)) cand = id = "_" + id;
   while (!ImageWindow.windowById(cand).isNull) cand = id + (++n);
   return cand;
}

function FD_newWindow(W, H, nc, color, id)
{
   return new ImageWindow(W, H, nc, 32, true, color, FD_uniqueId(id));
}

function FD_describe(P)
{
   var m = [];
   if (P.horizontal) m.push("horizontal bands");
   if (P.vertical) m.push("vertical bands");
   if (P.spikes) m.push("periodic spikes (" + P.spikeK + " sigma)");
   if (P.nyquist) m.push("checkerboard/Nyquist");
   var t = m.join(", ");
   if (P.adaptive && (P.horizontal || P.vertical)) t += "; adaptive amplitude matching (" + P.adaptWindow + " px)";
   return t;
}

function FD_makeProgress(verbose)
{
   return function(msg, frac)
   {
      if (verbose && frac < 0) console.writeln(msg);
      else if (verbose && frac === 0) console.writeln(msg + " ...");
      processEvents();
      var abort = false;
      try { abort = console.abortRequested; } catch (e) {}
      if (abort) throw new Error("FD_ABORT");
   };
}

// Process every channel of an Image. Returns { outs, noises, diag, rms[] }.
function FD_processImage(image, P, progress, allowDiag)
{
   var W = image.width, H = image.height, nc = image.numberOfChannels;
   var r = { outs: [], noises: [], diag: null, mask: null, gains: null, rms: [], spikes: 0 };
   for (var c = 0; c < nc; ++c)
   {
      progress("<b>Channel " + c + "</b>", -1);
      var data = FD_readChannel(image, c);
      var Pc = {};
      for (var k in P) Pc[k] = P[k];
      Pc.wantDiag = allowDiag && P.wantDiag && c == 0;
      Pc.wantMask = allowDiag && P.wantMask && c == 0;
      Pc.wantGains = allowDiag && P.wantGains && c == 0;
      var res = FD_processChannel(data, W, H, Pc, progress);
      r.outs.push(res.out);
      r.rms.push(res.rms);
      r.spikes += res.spikes;
      if (res.noise)
      {
         for (var i = 0; i < res.noise.length; ++i) res.noise[i] += 0.5;
         r.noises.push(res.noise);
      }
      if (res.diag) r.diag = res.diag;
      if (res.mask) r.mask = res.mask;
      if (res.gains) r.gains = res.gains;
      data = null;
      try { gc(); } catch (e) {}
   }
   return r;
}

function FD_historyKeyword(P)
{
   var txt = "FourierDeband " + FD_VERSION + ": " + FD_describe(P) +
             format(" sig=%.2f keep>%dpx str=%.2f", P.lineSigma, P.protectScale, P.strength) +
             (P.protectStars ? format(" mask=%.1f/%.1f/%d ref=%d", P.detectK, P.maskLow, P.maskGrow, P.refine) : " nostarprot") +
             ((P.adaptive && (P.horizontal || P.vertical)) ? format(" adapt=%dpx", P.adaptWindow) : "");
   var kw = [];
   for (var i = 0; i < txt.length; i += 70) kw.push(new FITSKeyword("HISTORY", "", txt.substring(i, i + 70)));
   return kw;
}

// ---------------------------------------------------------------------
// Single image
// ---------------------------------------------------------------------
function FD_run(view, P)
{
   var image = view.image, W = image.width, H = image.height, nc = image.numberOfChannels;
   var t0 = Date.now();
   console.show();
   try { console.abortEnabled = true; } catch (e) {}
   console.writeln("<end><cbr><br><b>FourierDeband</b>: " + view.id + " (" + W + " x " + H + ", " + nc + " channel" + (nc > 1 ? "s" : "") + ")");
   console.writeln("Targets: " + FD_describe(P));

   var r = FD_processImage(image, P, FD_makeProgress(true), true), c;

   var srcWin = view.window;
   if (P.inPlace)
   {
      view.beginProcess();
      for (c = 0; c < nc; ++c) FD_writeChannel(view.image, r.outs[c], c);
      view.endProcess();
   }
   else
   {
      var w = FD_newWindow(W, H, nc, image.isColor, view.id + "_deband");
      w.mainView.beginProcess(UndoFlag_NoSwapFile);
      for (c = 0; c < nc; ++c) FD_writeChannel(w.mainView.image, r.outs[c], c);
      w.mainView.endProcess();
      try { w.keywords = srcWin.keywords.concat(FD_historyKeyword(P)); } catch (e) {}
      try { w.copyAstrometricSolution(srcWin); } catch (e) {}
      w.show();
   }

   if (r.noises.length > 0)
   {
      var wn = FD_newWindow(W, H, nc, image.isColor, view.id + "_removed");
      wn.mainView.beginProcess(UndoFlag_NoSwapFile);
      for (c = 0; c < nc; ++c) FD_writeChannel(wn.mainView.image, r.noises[c], c);
      wn.mainView.endProcess();
      wn.show();
      console.writeln("Removed-noise image is offset by +0.5 (0.5 = nothing removed).");
   }

   if (r.diag)
   {
      var wd = FD_newWindow(W, H, 3, true, view.id + "_spectrum");
      wd.mainView.beginProcess(UndoFlag_NoSwapFile);
      FD_writeChannel(wd.mainView.image, r.diag.r, 0);
      FD_writeChannel(wd.mainView.image, r.diag.g, 1);
      FD_writeChannel(wd.mainView.image, r.diag.g, 2);
      wd.mainView.endProcess();
      wd.show();
      console.writeln("Spectrum: centered log|F| of the (star-protected, periodic) estimation image, channel 0. Red = attenuated.");
   }
   if (r.gains)
   {
      // gain maps, stored as gain/2 so that 0.5 = gain 1 (unchanged model), 0 = gain 0, 1 = gain 2
      var gnames = [["h", "_gain_rows"], ["v", "_gain_cols"]];
      for (var gq = 0; gq < 2; ++gq)
      {
         var gsrc = r.gains[gnames[gq][0]];
         if (!gsrc) continue;
         var gimg = new Float32Array(gsrc.length);
         for (var gi = 0; gi < gsrc.length; ++gi) gimg[gi] = 0.5 * gsrc[gi];
         var wg = FD_newWindow(W, H, 1, false, view.id + gnames[gq][1]);
         wg.mainView.beginProcess(UndoFlag_NoSwapFile);
         FD_writeChannel(wg.mainView.image, gimg, 0);
         wg.mainView.endProcess();
         wg.show();
      }
      console.writeln("Adaptive gain maps (channel 0): pixel value = gain / 2, so 0.5 = model used unchanged.");
   }
   if (r.mask)
   {
      var mf = new Float32Array(r.mask.length);
      for (var q = 0; q < mf.length; ++q) mf[q] = r.mask[q];
      var wm = FD_newWindow(W, H, 1, false, view.id + "_protection");
      wm.mainView.beginProcess(UndoFlag_NoSwapFile);
      FD_writeChannel(wm.mainView.image, mf, 0);
      wm.mainView.endProcess();
      wm.show();
      console.writeln("Protection mask (channel 0): white = excluded from the band estimate.");
   }
   for (c = 0; c < nc; ++c) console.writeln(format("Channel %d: RMS removed = %.3e", c, r.rms[c]));
   console.writeln(format("FourierDeband done in %.1f s", (Date.now() - t0) / 1000));
}

// ---------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------
var FD_WRITABLE = { ".xisf": 1, ".fit": 1, ".fits": 1, ".fts": 1 };

function FD_uniquePath(base, ext, overwrite)
{
   var path = base + ext;
   if (!overwrite)
      for (var n = 1; File.exists(path); ++n) path = base + "_" + n + ext;
   return path;
}

function FD_outputPath(inPath, P)
{
   var dir = P.outputDir.length > 0 ? P.outputDir : File.extractDrive(inPath) + File.extractDirectory(inPath);
   while (dir.length > 1 && (dir[dir.length - 1] == "/" || dir[dir.length - 1] == "\\")) dir = dir.substring(0, dir.length - 1);
   var ext = File.extractExtension(inPath).toLowerCase();
   if (!FD_WRITABLE[ext]) ext = ".xisf";          // raw camera files etc.
   return FD_uniquePath(dir + "/" + File.extractName(inPath) + P.postfix, ext, P.overwrite);
}

function FD_saveWindow(win, path)
{
   // saveAs(filePath, queryOptions, allowMessages, strict, verifyOverwrite)
   if (!win.saveAs(path, false, false, false, false))
      throw new Error("cannot write " + path);
}

function FD_runBatch(files, P)
{
   var t0 = Date.now(), ok = 0, fail = [], table = [];
   console.show();
   try { console.abortEnabled = true; } catch (e) {}
   console.writeln("<end><cbr><br><b>FourierDeband batch</b>: " + files.length + " file(s)");
   console.writeln("Targets: " + FD_describe(P));
   console.writeln("Output: " + (P.outputDir.length ? P.outputDir : "(same folder as each input)") + ", postfix '" + P.postfix + "'");
   if (P.outputDir.length && !File.directoryExists(P.outputDir))
      File.createDirectory(P.outputDir, true);

   var progress = FD_makeProgress(false);
   var aborted = false;
   for (var f = 0; f < files.length && !aborted; ++f)
   {
      var path = files[f], t1 = Date.now(), wins = null;
      console.writeln(format("<br>[%d/%d] ", f + 1, files.length) + path);
      processEvents();
      try
      {
         if (!File.exists(path)) throw new Error("file not found");
         wins = ImageWindow.open(path);
         if (!wins || wins.length == 0 || wins[0].isNull) throw new Error("cannot open");
         for (var wi = 1; wi < wins.length; ++wi) wins[wi].forceClose();   // only the first image of multi-image files
         var win = wins[0], view = win.mainView, img = view.image;
         var nc = img.numberOfChannels;

         var r = FD_processImage(img, P, progress, false);

         // write into the opened window itself: keeps keywords, XISF properties, astrometry, ICC, etc.
         view.beginProcess(UndoFlag_NoSwapFile);
         for (var c = 0; c < nc; ++c) FD_writeChannel(view.image, r.outs[c], c);
         view.endProcess();
         try { win.keywords = win.keywords.concat(FD_historyKeyword(P)); } catch (e) {}
         var outPath = FD_outputPath(path, P);
         FD_saveWindow(win, outPath);

         if (r.noises.length > 0)
         {
            var wn = FD_newWindow(img.width, img.height, nc, img.isColor, "FD_noise_tmp");
            wn.mainView.beginProcess(UndoFlag_NoSwapFile);
            for (c = 0; c < nc; ++c) FD_writeChannel(wn.mainView.image, r.noises[c], c);
            wn.mainView.endProcess();
            var oext = File.extractExtension(outPath);
            FD_saveWindow(wn, FD_uniquePath(outPath.substring(0, outPath.length - oext.length) + "_removed", oext, P.overwrite));
            wn.forceClose();
         }
         var rsum = 0;
         for (c = 0; c < nc; ++c) rsum += r.rms[c];
         r = null;
         console.writeln("  -> " + outPath + format("  (%.1f s)", (Date.now() - t1) / 1000));
         table.push({ name: File.extractNameAndExtension(path), rms: rsum / nc });
         ++ok;
      }
      catch (e)
      {
         if (e.message == "FD_ABORT") { aborted = true; console.warningln("*** Aborted by user"); }
         else { fail.push(path); console.criticalln("  *** Error: " + e.message); }
      }
      if (wins) for (var q = 0; q < wins.length; ++q) try { if (!wins[q].isNull) wins[q].forceClose(); } catch (e) {}
      try { gc(); } catch (e) {}
   }

   // summary
   console.writeln("<br><b>FourierDeband batch summary</b>");
   if (table.length > 0)
   {
      // RMS of what was removed, relative to the batch median: outliers are subs with unusual pattern noise
      var vals = [];
      for (var i = 0; i < table.length; ++i) vals.push(table[i].rms);
      var sorted = vals.slice().sort(function(a, b) { return a - b; });
      var med = sorted[sorted.length >> 1];
      console.writeln("RMS removed (channel mean) per file, x = ratio to batch median " + format("%.3e", med) + ":");
      for (i = 0; i < table.length; ++i)
      {
         var ratio = med > 0 ? table[i].rms / med : 0;
         var line = format("  %.3e  x%.2f  ", table[i].rms, ratio) + table[i].name;
         if (ratio > 2) console.warningln(line + "   <- unusually strong pattern noise"); else console.writeln(line);
      }
   }
   console.writeln(format("%d succeeded, %d failed%s, %.1f s total", ok, fail.length, aborted ? ", aborted" : "", (Date.now() - t0) / 1000));
   for (i = 0; i < fail.length; ++i) console.criticalln("  failed: " + fail[i]);
}

// ---------------------------------------------------------------------
// Dialog
// ---------------------------------------------------------------------
function FD_Dialog(P)
{
   this.__base__ = Dialog;
   this.__base__();
   var self = this;
   // Handlers stay inert while the dialog is being built and after it closes:
   // PJSR/Qt can deliver toggle events from initial-state assignments.
   var live = false;
   this.kill = function() { live = false; };
   function updateEnable()
   {
      if (!self.cbDiag || !self.gBatch) return;           // not fully built yet
      self.ncWidth.enabled = self.ncProtect.enabled = self.cbAdapt.enabled = P.horizontal || P.vertical;
      self.ncAdaptW.enabled = (P.horizontal || P.vertical) && P.adaptive;
      self.cbGains.enabled = (P.horizontal || P.vertical) && P.adaptive && !P.batch;
      self.ncK.enabled = self.ncDil.enabled = P.spikes;
      self.ncNyq.enabled = P.nyquist;
      self.ncDetect.enabled = self.ncLow.enabled = self.ncGrow.enabled = self.ncRefine.enabled = P.protectStars;
      self.cbMask.enabled = P.protectStars && !P.batch;
      self.viewList.enabled = !P.batch;
      self.rbNew.enabled = self.rbIn.enabled = !P.batch;
      self.cbDiag.enabled = !P.batch;
      self.gBatch.enabled = P.batch;
   }
   var labelW = this.font.width("Keep structures larger than (px):") + 8;
   var editW = this.font.width("00000.00");

   function numeric(parent, label, lo, hi, prec, key, tip)
   {
      var n = new NumericControl(parent);
      n.label.text = label;
      n.label.minWidth = labelW;
      n.setRange(lo, hi);
      n.slider.setRange(0, 1000);
      n.slider.minWidth = 200;
      n.setPrecision(prec);
      n.edit.minWidth = editW;
      n.setValue(P[key]);
      n.toolTip = tip;
      n.onValueUpdated = function(v) { P[key] = (prec == 0) ? Math.round(v) : v; };
      return n;
   }
   function check(parent, text, key, tip)
   {
      var cb = new CheckBox(parent);
      cb.text = text;
      cb.checked = P[key];
      cb.toolTip = tip;
      cb.onCheck = function(v) { if (!live) return; P[key] = v; updateEnable(); };
      return cb;
   }
   function group(title, items)
   {
      var g = new GroupBox(self);
      g.title = title;
      g.sizer = new VerticalSizer;
      g.sizer.margin = 6;
      g.sizer.spacing = 4;
      for (var i = 0; i < items.length; ++i) g.sizer.add(items[i]);
      return g;
   }

   this.help = new Label(this);
   this.help.frameStyle = FrameStyle_Box;
   this.help.margin = 6;
   this.help.wordWrapping = true;
   this.help.useRichText = true;
   this.help.minWidth = 520;
   this.help.text = "<b>FourierDeband</b> removes CMOS row/column banding and other periodic noise " +
      "by notching them in the 2-D Fourier spectrum. Horizontal bands (row offsets) live on the vertical " +
      "axis of the spectrum, vertical bands on the horizontal axis; low frequencies are protected so gradients " +
      "and nebulae are untouched. The noise is estimated on a copy with stars masked out and subtracted from the original. " +
      "Best on linear data. Batch mode is meant for calibrated subs <i>before</i> registration, where the " +
      "banding is still aligned with the pixel grid.";

   // ---- mode
   // Each radio pair gets its own parent: Qt makes sibling radio buttons
   // mutually exclusive, which otherwise couples mode and output choices.
   this.modeBox = new Control(this);
   this.rbSingle = new RadioButton(this.modeBox); this.rbSingle.text = "Image:";
   this.rbSingle.checked = !P.batch;
   this.viewList = new ViewList(this.modeBox);
   this.viewList.getMainViews();
   if (ImageWindow.activeWindow && !ImageWindow.activeWindow.isNull)
      this.viewList.currentView = ImageWindow.activeWindow.mainView;
   this.rbBatch = new RadioButton(this.modeBox); this.rbBatch.text = "Batch files (below)";
   this.rbBatch.checked = P.batch;
   this.rbSingle.checked = !P.batch;
   this.rbSingle.onCheck = function(v) { if (!live) return; P.batch = !v; updateEnable(); };
   this.rbBatch.onCheck = function(v) { if (!live) return; P.batch = v; updateEnable(); };
   this.modeBox.sizer = new HorizontalSizer;
   this.modeBox.sizer.spacing = 6;
   this.modeBox.sizer.add(this.rbSingle); this.modeBox.sizer.add(this.viewList, 100);
   this.modeBox.sizer.addSpacing(12); this.modeBox.sizer.add(this.rbBatch);
   this.targetSizer = new HorizontalSizer;
   this.targetSizer.add(this.modeBox, 100);

   // ---- banding
   this.cbH = check(this, "Horizontal bands (row-to-row offsets)", "horizontal",
      "Removes the vertical line through the spectrum center (fx = 0), i.e. offsets that are constant along each row.");
   this.cbV = check(this, "Vertical bands (column-to-column offsets)", "vertical",
      "Removes the horizontal line through the spectrum center (fy = 0), i.e. offsets that are constant along each column.");
   this.ncWidth = numeric(this, "Notch width sigma (bins):", 0.3, 5, 2, "lineSigma",
      "Gaussian half-width of the axis notch in frequency bins. 0.5-1 for bands that span the full sensor width; " +
      "larger if the band amplitude varies across the frame (partial-width bands).");
   this.ncProtect = numeric(this, "Keep structures larger than (px):", 8, 2048, 0, "protectScale",
      "Band components with a period longer than this are left alone (treated as gradient, for DBE/ABE). " +
      "Full notch from half this period down to 2 px, raised-cosine taper in between.");
   this.cbAdapt = check(this, "Adaptive amplitude matching (local gain per band component)", "adaptive",
      "Fits a smoothly varying gain for the row-band and the column-band models, by windowed least squares on the " +
      "unprotected pixels (as in seismic adaptive subtraction). Corrects bands whose strength varies across the sensor " +
      "(fading bands, different readout halves) and the under-correction of rows crossing large protected areas. " +
      "Gains shrink towards 1 where the bands are too weak to measure, so frames without banding are left as without " +
      "the option. Costs one extra inverse FFT per band direction.");
   this.ncAdaptW = numeric(this, "Adaptive window (px):", 32, 512, 0, "adaptWindow",
      "Size of the square window over which each local gain is fitted. Smaller follows faster changes of band " +
      "strength but is noisier; 64-256 is a sensible range.");
   this.cbGains = check(this, "Output gain maps (single image)", "wantGains",
      "Shows the fitted row-band and column-band gains (pixel value = gain / 2; 0.5 means the model was used unchanged).");
   this.gBand = group("Banding", [this.cbH, this.cbV, this.ncWidth, this.ncProtect, this.cbAdapt, this.ncAdaptW, this.cbGains]);

   // ---- other periodic noise
   this.cbSpikes = check(this, "Periodic pattern noise (auto-detect spectral spikes)", "spikes",
      "Finds off-axis bins whose magnitude stands above the local spectral background (block median of log|F|, " +
      "robust sigma) and pulls them back to that background. Targets electronic interference, fixed-pattern grids, " +
      "moire and drizzle/registration lattices.");
   this.ncK = numeric(this, "Spike threshold (sigma):", 3, 20, 1, "spikeK",
      "Detection threshold in robust sigmas of the log-magnitude spectrum. White noise gives no detections at 6.");
   this.ncDil = numeric(this, "Spike dilation (bins):", 0, 4, 0, "spikeDilate",
      "Also attenuate neighbouring bins that sit above background (spectral leakage around off-grid frequencies).");
   this.cbNyq = check(this, "Checkerboard / Nyquist pattern", "nyquist",
      "Notches the Nyquist row and column of the spectrum: even/odd pixel alternation from debayering, 2x drizzle, " +
      "or dual-readout sensors.");
   this.ncNyq = numeric(this, "Nyquist notch width (bins):", 0.5, 5, 2, "nyquistSigma", "Gaussian width of the Nyquist notches.");
   this.gOther = group("Other Fourier-domain noise", [this.cbSpikes, this.ncK, this.ncDil, this.cbNyq, this.ncNyq]);

   // ---- protection
   this.cbStars = check(this, "Protect stars and bright structures (mask)", "protectStars",
      "Masks stars with their halos, diffraction spikes, saturated cores and other compact bright or dark structures, " +
      "fills them from a smooth background, and estimates the bands from the unmasked pixels only. Without it a bright " +
      "star is subtracted along its whole row and column (black streak). Strongly recommended.");
   this.ncDetect = numeric(this, "Detection (sigma, smoothed):", 2, 20, 1, "detectK",
      "Seed threshold on the 5x5-smoothed residual above the background, in sigmas of that smoothed residual. " +
      "5 catches stars down to about 3 sigma per-pixel peak with essentially no false detections in pure noise. " +
      "Saturated pixels are always seeds.");
   this.ncLow = numeric(this, "Halo threshold (sigma, smoothed):", 0.5, 5, 2, "maskLow",
      "The mask grows from each seed while the smoothed residual stays above this level: it follows halos and spikes " +
      "down to the noise floor. Lower = larger masks around bright stars.");
   this.ncGrow = numeric(this, "Mask dilation (px):", 0, 64, 0, "maskGrow", "Final dilation of the mask, for the faint outer wings.");
   this.ncRefine = numeric(this, "Refinement passes:", 0, 4, 0, "refine",
      "Extra passes in which masked pixels are refilled with background + current band model (POCS). Each pass costs one " +
      "more forward+inverse FFT; 1 is enough unless large areas are masked.");
   this.cbMask = check(this, "Output protection mask (single image)", "wantMask", "Shows the mask actually used (white = protected).");
   this.gProt = group("Protection", [this.cbStars, this.ncDetect, this.ncLow, this.ncGrow, this.ncRefine, this.cbMask]);

   // ---- output
   this.ncStr = numeric(this, "Strength:", 0, 1, 2, "strength", "Fraction of the modelled noise that is removed.");
   this.outBox = new Control(this);
   this.rbNew = new RadioButton(this.outBox); this.rbNew.text = "Create new image";
   this.rbIn = new RadioButton(this.outBox); this.rbIn.text = "Apply to target (can be undone)";
   this.rbIn.checked = P.inPlace; this.rbNew.checked = !P.inPlace;
   this.rbNew.onCheck = function(v) { if (!live) return; P.inPlace = !v; };
   this.rbIn.onCheck = function(v) { if (!live) return; P.inPlace = v; };
   this.outBox.sizer = new HorizontalSizer;
   this.outBox.sizer.add(this.rbNew); this.outBox.sizer.addSpacing(12); this.outBox.sizer.add(this.rbIn); this.outBox.sizer.addStretch();
   var rbs = this.outBox;
   this.cbNoise = check(this, "Output removed-noise image", "wantNoise",
      "Shows exactly what was subtracted (offset +0.5). In batch mode it is saved next to each output as <name><postfix>_removed.");
   this.cbDiag = check(this, "Output spectrum diagnostic (channel 0, single image only)", "wantDiag",
      "Centered log-magnitude spectrum with attenuated regions tinted red. Use it on one representative sub to tune the settings before a batch run.");
   this.gOut = group("Output", [this.ncStr, rbs, this.cbNoise, this.cbDiag]);

   // ---- batch
   this.files = new TreeBox(this);
   this.files.rootDecoration = false;
   this.files.alternateRowColor = true;
   this.files.multipleSelection = true;
   this.files.headerVisible = false;
   this.files.numberOfColumns = 1;
   this.files.setScaledMinSize(520, 140);
   var initial = P.fileList.length ? P.fileList.split("|") : [];
   for (var i = 0; i < initial.length; ++i) { var nd = new TreeBoxNode(this.files); nd.setText(0, initial[i]); }

   this.syncFiles = function()
   {
      var a = [];
      for (var i = 0; i < self.files.numberOfChildren; ++i) a.push(self.files.child(i).text(0));
      P.fileList = a.join("|");
      self.fileCount.text = a.length + " file(s)";
   };
   this.btAdd = new PushButton(this); this.btAdd.text = "Add files";
   this.btAdd.onClick = function()
   {
      var d = new OpenFileDialog;
      d.multipleSelections = true;
      d.caption = "FourierDeband: select images";
      d.filters = [["Images", "*.xisf", "*.fit", "*.fits", "*.fts"], ["All files", "*"]];
      if (d.execute())
      {
         var have = {};
         for (var i = 0; i < self.files.numberOfChildren; ++i) have[self.files.child(i).text(0)] = 1;
         for (i = 0; i < d.fileNames.length; ++i)
            if (!have[d.fileNames[i]]) { var n = new TreeBoxNode(self.files); n.setText(0, d.fileNames[i]); }
         self.syncFiles();
      }
   };
   this.btRemove = new PushButton(this); this.btRemove.text = "Remove selected";
   this.btRemove.onClick = function()
   {
      for (var i = self.files.numberOfChildren - 1; i >= 0; --i)
         if (self.files.child(i).selected) self.files.remove(i);
      self.syncFiles();
   };
   this.btClear = new PushButton(this); this.btClear.text = "Clear";
   this.btClear.onClick = function() { self.files.clear(); self.syncFiles(); };
   this.fileCount = new Label(this);
   this.fileCount.textAlignment = TextAlign_Right | TextAlign_VertCenter;
   var fbs = new HorizontalSizer; fbs.spacing = 6;
   fbs.add(this.btAdd); fbs.add(this.btRemove); fbs.add(this.btClear); fbs.addStretch(); fbs.add(this.fileCount);

   this.outDirEdit = new Edit(this);
   this.outDirEdit.text = P.outputDir;
   this.outDirEdit.toolTip = "Leave empty to write each output next to its input file.";
   this.outDirEdit.onEditCompleted = function() { P.outputDir = this.text.trim(); };
   this.btDir = new ToolButton(this);
   this.btDir.icon = this.scaledResource(":/browser/select-file.png");
   this.btDir.setScaledFixedSize(22, 22);
   this.btDir.toolTip = "Select output folder";
   this.btDir.onClick = function()
   {
      var d = new GetDirectoryDialog;
      d.caption = "FourierDeband: output folder";
      if (P.outputDir.length) d.initialPath = P.outputDir;
      if (d.execute()) { P.outputDir = d.directory; self.outDirEdit.text = d.directory; }
   };
   var ol = new Label(this); ol.text = "Output folder:"; ol.textAlignment = TextAlign_Right | TextAlign_VertCenter;
   var ods = new HorizontalSizer; ods.spacing = 6; ods.add(ol); ods.add(this.outDirEdit, 100); ods.add(this.btDir);

   this.postEdit = new Edit(this);
   this.postEdit.text = P.postfix;
   this.postEdit.setFixedWidth(this.font.width("MMMMMMMM"));
   this.postEdit.toolTip = "Appended to each file name. The extension is kept (XISF/FITS); other formats are written as XISF.";
   this.postEdit.onEditCompleted = function() { P.postfix = this.text.trim(); };
   var pl = new Label(this); pl.text = "Postfix:"; pl.textAlignment = TextAlign_Right | TextAlign_VertCenter;
   this.cbOver = check(this, "Overwrite existing outputs", "overwrite",
      "If unchecked, an existing output gets a _1, _2 ... suffix instead of being replaced.");
   var pps = new HorizontalSizer; pps.spacing = 6; pps.add(pl); pps.add(this.postEdit); pps.addSpacing(16); pps.add(this.cbOver); pps.addStretch();

   this.gBatch = group("Batch files", [this.files, fbs, ods, pps]);
   this.syncFiles();

   // ---- buttons
   this.btReset = new PushButton(this); this.btReset.text = "Reset";
   this.btReset.onClick = function()
   {
      var d = FD_defaults();
      for (var k in d) if (k != "fileList" && k != "outputDir") P[k] = d[k];
      self.ok(); self.resetRequested = true;
   };
   this.btOK = new PushButton(this); this.btOK.text = "Execute"; this.btOK.icon = this.scaledResource(":/icons/ok.png");
   this.btOK.onClick = function()
   {
      P.outputDir = self.outDirEdit.text.trim();
      P.postfix = self.postEdit.text.trim();
      P.batch = self.rbBatch.checked;
      P.inPlace = self.rbIn.checked;
      if (!P.horizontal && !P.vertical && !P.spikes && !P.nyquist)
      { (new MessageBox("Nothing selected to remove.", "FourierDeband", StdIcon_Warning, StdButton_Ok)).execute(); return; }
      if (P.batch)
      {
         if (self.files.numberOfChildren == 0)
         { (new MessageBox("Add some files to the batch list.", "FourierDeband", StdIcon_Warning, StdButton_Ok)).execute(); return; }
         if (P.postfix.length == 0 && P.outputDir.length == 0 && P.overwrite)
         {
            if ((new MessageBox("Empty postfix, no output folder and overwrite on: the input files will be replaced. Continue?",
                  "FourierDeband", StdIcon_Warning, StdButton_Yes, StdButton_No)).execute() != StdButton_Yes) return;
         }
      }
      else if (self.viewList.currentView.isNull)
      { (new MessageBox("Select a target image.", "FourierDeband", StdIcon_Warning, StdButton_Ok)).execute(); return; }
      self.ok();
   };
   this.btCancel = new PushButton(this); this.btCancel.text = "Cancel"; this.btCancel.icon = this.scaledResource(":/icons/cancel.png");
   this.btCancel.onClick = function() { self.cancel(); };
   var bs = new HorizontalSizer; bs.spacing = 6; bs.add(this.btReset); bs.addStretch(); bs.add(this.btOK); bs.add(this.btCancel);

   this.updateEnable = updateEnable;
   live = true;
   updateEnable();

   this.sizer = new VerticalSizer;
   this.sizer.margin = 8;
   this.sizer.spacing = 6;
   this.sizer.add(this.help);
   this.sizer.add(this.targetSizer);
   this.sizer.add(this.gBand);
   this.sizer.add(this.gOther);
   this.sizer.add(this.gProt);
   this.sizer.add(this.gOut);
   this.sizer.add(this.gBatch, 100);
   this.sizer.add(bs);
   this.windowTitle = "FourierDeband " + FD_VERSION;
   this.adjustToContents();
   this.setMinSize();
}
FD_Dialog.prototype = new Dialog;

function main()
{
   var P = FD_defaults();
   FD_loadSettings(P);
   for (;;)
   {
      var d = new FD_Dialog(P);
      d.resetRequested = false;
      var accepted = d.execute();
      d.kill();
      if (!accepted) { FD_saveSettings(P); return; }
      if (d.resetRequested) continue;
      FD_saveSettings(P);
      try
      {
         if (P.batch) FD_runBatch(P.fileList.split("|"), P);
         else FD_run(d.viewList.currentView, P);
      }
      catch (e)
      {
         if (e.message == "FD_ABORT") console.warningln("FourierDeband: aborted by user");
         else console.criticalln("FourierDeband: " + e.message);
      }
      return;
   }
}

main();
