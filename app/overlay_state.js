(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.ScreenCensorOverlayState = api;
})(typeof globalThis !== "undefined" ? globalThis : window, function () {
  const DEFAULT_BEHAVIOR = {
    smooth: 0.42,
    predict: 0.05,
    holdMs: 260,
    fadeInMs: 120,
    fadeOutMs: 220,
    matchPx: 180,
    confirmFrames: 2,
    strongConfirmFrames: 3,
    candidateBlend: 0.58,
    promoteBlend: 0.34,
    maxPredictMs: 120,
    stableAcceptScore: 0.52,
    candidateAcceptScore: 0.62,
    maxAreaRatio: 3.0
  };

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function clonePoints(points) {
    if (!Array.isArray(points)) return [];
    return points.map((point) => ({
      x: Number(point.x) || 0,
      y: Number(point.y) || 0
    }));
  }

  function pointsCompatible(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length >= 3 && a.length === b.length;
  }

  function centroid(points) {
    if (!Array.isArray(points) || points.length === 0) {
      return { x: 0, y: 0 };
    }

    let sx = 0;
    let sy = 0;

    for (const point of points) {
      sx += Number(point.x) || 0;
      sy += Number(point.y) || 0;
    }

    return {
      x: sx / points.length,
      y: sy / points.length
    };
  }

  function bbox(points) {
    if (!Array.isArray(points) || points.length === 0) {
      return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0, diagonal: 0 };
    }

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const point of points) {
      const x = Number(point.x) || 0;
      const y = Number(point.y) || 0;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }

    const width = Math.max(0, maxX - minX);
    const height = Math.max(0, maxY - minY);

    return {
      minX,
      minY,
      maxX,
      maxY,
      width,
      height,
      diagonal: Math.hypot(width, height)
    };
  }

  function rectPolygonFromBox(box, padX, padY, minWidth = 0, minHeight = 0) {
    const cx = (box.minX + box.maxX) * 0.5;
    const cy = (box.minY + box.maxY) * 0.5;
    const width = Math.max(minWidth, box.width + padX * 2);
    const height = Math.max(minHeight, box.height + padY * 2);
    const halfW = width * 0.5;
    const halfH = height * 0.5;

    return [
      { x: cx - halfW, y: cy - halfH },
      { x: cx + halfW, y: cy - halfH },
      { x: cx + halfW, y: cy + halfH },
      { x: cx - halfW, y: cy + halfH }
    ];
  }

  function boxContainsPoint(box, point, pad = 0) {
    if (!box || !point) return false;
    return (
      point.x >= box.minX - pad &&
      point.x <= box.maxX + pad &&
      point.y >= box.minY - pad &&
      point.y <= box.maxY + pad
    );
  }

  function spatialBucketKey(points, size = 48) {
    const c = centroid(points);
    return `bucket:${Math.round(c.x / size)}:${Math.round(c.y / size)}`;
  }

  function findBestFaceAnchor(detection, faceAnchors) {
    if (!detection || !Array.isArray(faceAnchors) || faceAnchors.length === 0) {
      return null;
    }

    const eyeCenter = centroid(detection.points);
    const eyeBox = bbox(detection.points);
    let bestFace = null;
    let bestScore = Number.POSITIVE_INFINITY;

    for (const face of faceAnchors) {
      if (!face?.box) continue;

      const facePad = Math.max(12, face.box.diagonal * 0.08);
      const withinFace = boxContainsPoint(face.box, eyeCenter, facePad);
      const horizontalOverlap =
        eyeBox.maxX >= face.box.minX - facePad && eyeBox.minX <= face.box.maxX + facePad;
      const verticalWindow =
        eyeCenter.y >= face.box.minY - facePad &&
        eyeCenter.y <= face.box.minY + face.box.height * 0.72;
      if (!withinFace && !(horizontalOverlap && verticalWindow)) {
        continue;
      }

      const score =
        distance(eyeCenter, face.center) / Math.max(40, face.box.diagonal) +
        Math.max(0, eyeCenter.y - face.box.minY) / Math.max(1, face.box.height) * 0.18;
      if (score < bestScore) {
        bestScore = score;
        bestFace = face;
      }
    }

    return bestFace;
  }

  function polygonArea(points) {
    if (!Array.isArray(points) || points.length < 3) {
      return 0;
    }

    let area = 0;

    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      area += (Number(a.x) || 0) * (Number(b.y) || 0) - (Number(b.x) || 0) * (Number(a.y) || 0);
    }

    return Math.abs(area) * 0.5;
  }

  function distance(a, b) {
    return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0));
  }

  function meanPointDistance(a, b) {
    if (!pointsCompatible(a, b)) {
      return Number.POSITIVE_INFINITY;
    }

    let total = 0;
    for (let i = 0; i < a.length; i++) {
      total += distance(a[i], b[i]);
    }
    return total / Math.max(1, a.length);
  }

  function shapeDifferenceScore(a, b) {
    if (!pointsCompatible(a, b)) {
      return Number.POSITIVE_INFINITY;
    }

    const centerDistance = distance(centroid(a), centroid(b));
    const pointDistance = meanPointDistance(a, b);
    const boxA = bbox(a);
    const boxB = bbox(b);
    const areaA = polygonArea(a);
    const areaB = polygonArea(b);
    const scale = Math.max(40, boxA.diagonal, boxB.diagonal);
    const areaRatio =
      areaA > 0 && areaB > 0
        ? Math.max(areaA, areaB) / Math.max(1, Math.min(areaA, areaB))
        : 1;
    const areaPenalty = Math.max(0, areaRatio - 1);

    return centerDistance / scale + pointDistance / scale * 0.9 + areaPenalty * 0.22;
  }

  function lerpPoints(fromPoints, toPoints, alpha) {
    if (!pointsCompatible(fromPoints, toPoints)) {
      return clonePoints(toPoints);
    }

    const t = clamp01(alpha);
    const out = [];

    for (let i = 0; i < toPoints.length; i++) {
      out.push({
        x: fromPoints[i].x * (1 - t) + toPoints[i].x * t,
        y: fromPoints[i].y * (1 - t) + toPoints[i].y * t
      });
    }

    return out;
  }

  function smoothPolygon(prevPoints, nextPoints, alpha) {
    if (!pointsCompatible(prevPoints, nextPoints)) {
      return clonePoints(nextPoints);
    }

    return lerpPoints(prevPoints, nextPoints, clamp01(alpha));
  }

  function zeroVelocity(points) {
    if (!Array.isArray(points)) return [];
    return points.map(() => ({ x: 0, y: 0 }));
  }

  function computeVelocity(prevPoints, nextPoints, prevVelocity, deltaMs) {
    if (!pointsCompatible(prevPoints, nextPoints)) {
      return zeroVelocity(nextPoints);
    }

    const dt = Math.max(1, deltaMs || 16);
    const carry = 0.72;
    const out = [];

    for (let i = 0; i < nextPoints.length; i++) {
      const prevV = prevVelocity?.[i] || { x: 0, y: 0 };
      const vx = (nextPoints[i].x - prevPoints[i].x) / dt;
      const vy = (nextPoints[i].y - prevPoints[i].y) / dt;
      out.push({
        x: prevV.x * carry + vx * (1 - carry),
        y: prevV.y * carry + vy * (1 - carry)
      });
    }

    return out;
  }

  function predictPolygon(points, velocity, predictStrength, aheadMs) {
    if (!pointsCompatible(points, velocity)) {
      return clonePoints(points);
    }

    const scale = Math.max(0, predictStrength || 0) * Math.max(0, aheadMs || 0);
    if (scale <= 0) {
      return clonePoints(points);
    }

    return points.map((point, index) => ({
      x: point.x + velocity[index].x * scale,
      y: point.y + velocity[index].y * scale
    }));
  }

  function behaviorFor(partBehavior, part) {
    return {
      ...DEFAULT_BEHAVIOR,
      ...(partBehavior?.[part] || {})
    };
  }

  function normalizeDetection(poly) {
    if (!poly || !poly.id || !poly.part || !Array.isArray(poly.points) || poly.points.length < 3) {
      return null;
    }

    return {
      id: String(poly.id),
      part: String(poly.part),
      personId: poly.person_id == null ? null : String(poly.person_id),
      points: clonePoints(poly.points)
    };
  }

  function mergeEyesDetections(detections) {
    const passthrough = [];
    const eyeGroups = new Map();
    const faceAnchors = [];

    for (const detection of detections || []) {
      if (detection?.part === "face") {
        faceAnchors.push({
          id: detection.id,
          personId: detection.personId,
          center: centroid(detection.points),
          box: bbox(detection.points)
        });
      }
    }

    for (const detection of detections || []) {
      if (!detection || detection.part !== "eyes") {
        passthrough.push(detection);
        continue;
      }

      const faceAnchor = findBestFaceAnchor(detection, faceAnchors);
      const groupKey = faceAnchor
        ? faceAnchor.personId != null
          ? `face-person:${faceAnchor.personId}`
          : `face-id:${faceAnchor.id}`
        : detection.personId != null
          ? `person:${detection.personId}`
          : spatialBucketKey(detection.points);
      if (!eyeGroups.has(groupKey)) {
        eyeGroups.set(groupKey, {
          key: groupKey,
          faceAnchor,
          items: []
        });
      }
      const group = eyeGroups.get(groupKey);
      if (faceAnchor && !group.faceAnchor) {
        group.faceAnchor = faceAnchor;
      }
      group.items.push(detection);
    }

    for (const group of eyeGroups.values()) {
      if (!group || group.items.length === 0) continue;

      let mergedBox = null;
      for (const detection of group.items) {
        const box = bbox(detection.points);
        if (!mergedBox) {
          mergedBox = { ...box };
          continue;
        }
        mergedBox.minX = Math.min(mergedBox.minX, box.minX);
        mergedBox.minY = Math.min(mergedBox.minY, box.minY);
        mergedBox.maxX = Math.max(mergedBox.maxX, box.maxX);
        mergedBox.maxY = Math.max(mergedBox.maxY, box.maxY);
        mergedBox.width = mergedBox.maxX - mergedBox.minX;
        mergedBox.height = mergedBox.maxY - mergedBox.minY;
        mergedBox.diagonal = Math.hypot(mergedBox.width, mergedBox.height);
      }

      const groupDiag = Math.max(18, mergedBox?.diagonal || 0);
      const padX = Math.max(10, (mergedBox?.width || 0) * 0.42, groupDiag * 0.26);
      const padY = Math.max(8, (mergedBox?.height || 0) * 1.4, groupDiag * 0.18);
      const rectPoints = rectPolygonFromBox(
        mergedBox || { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 },
        padX,
        padY,
        Math.max(34, (mergedBox?.width || 0) * 1.9),
        Math.max(16, (mergedBox?.height || 0) * 3.2)
      );
      const stableEyeId = group.faceAnchor
        ? `eyes_block_${group.key}`
        : group.items[0].personId != null
          ? `eyes_block_person:${group.items[0].personId}`
          : `eyes_block_${spatialBucketKey(rectPoints, 56)}`;
      const stablePersonId = group.faceAnchor?.personId ?? group.items[0].personId ?? null;

      passthrough.push({
        id: stableEyeId,
        part: "eyes",
        personId: stablePersonId,
        points: rectPoints
      });
    }

    return passthrough;
  }

  function createLegacyOverlayState(options = {}) {
    const trackedPolygons = new Map();
    const partBehavior = options.partBehavior || {};
    const defaultHoldFrames = Number(options.defaultHoldFrames || 6);

    function clear() {
      trackedPolygons.clear();
    }

    function ingest(rawPolygons) {
      const seen = new Set();

      for (const rawPoly of rawPolygons || []) {
        const poly = normalizeDetection(rawPoly);
        if (!poly) continue;

        seen.add(poly.id);
        const prev = trackedPolygons.get(poly.id);
        const behavior = behaviorFor(partBehavior, poly.part);

        if (!prev) {
          trackedPolygons.set(poly.id, {
            id: poly.id,
            part: poly.part,
            personId: poly.personId,
            points: clonePoints(poly.points),
            velocity: zeroVelocity(poly.points),
            missing: 0
          });
          continue;
        }

        const smoothed = smoothPolygon(prev.points, poly.points, behavior.smooth);
        const velocity = computeVelocity(prev.points, smoothed, prev.velocity, 16);

        trackedPolygons.set(poly.id, {
          id: poly.id,
          part: poly.part,
          personId: poly.personId,
          points: smoothed,
          velocity,
          missing: 0
        });
      }

      for (const [id, value] of trackedPolygons.entries()) {
        if (!seen.has(id)) {
          value.missing = (value.missing || 0) + 1;
          if (value.missing > defaultHoldFrames) {
            trackedPolygons.delete(id);
          }
        }
      }
    }

    function advance() {
      const out = [];

      for (const item of trackedPolygons.values()) {
        const behavior = behaviorFor(partBehavior, item.part);
        out.push({
          id: item.id,
          stableId: item.id,
          part: item.part,
          person_id: item.personId,
          alpha: 1,
          points: predictPolygon(item.points, item.velocity, behavior.predict * 16, 1)
        });
      }

      return out;
    }

    return {
      clear,
      ingest,
      advance
    };
  }

  function createStableOverlayState(options = {}) {
    const tracks = new Map();
    const partBehavior = options.partBehavior || {};
    let nextTrackId = 1;

    function clear(hard = true) {
      if (hard) {
        tracks.clear();
        return;
      }

      for (const track of tracks.values()) {
        track.targetAlpha = 0;
      }
    }

    function matchScore(track, detection) {
      if (track.part !== detection.part) return Number.POSITIVE_INFINITY;

      const behavior = behaviorFor(partBehavior, detection.part);
      const detectionCenter = centroid(detection.points);
      const stablePoints = track.lastStablePoints || track.renderPoints || track.lastSeenPoints || track.targetPoints;
      const trackCenter = centroid(stablePoints);
      const trackBox = bbox(stablePoints);
      const detectionBox = bbox(detection.points);
      const trackArea = polygonArea(stablePoints);
      const detectionArea = polygonArea(detection.points);
      const missingForMs =
        track.missingSince == null || track.lastTouchedAt == null ? 0 : Math.max(0, track.lastTouchedAt - track.missingSince);
      const staleFactor = track.missCount > 0 ? Math.max(0.55, 1 - Math.min(0.45, track.missCount * 0.08)) : 1;
      const maxDistance = Math.max(
        behavior.matchPx * staleFactor,
        trackBox.diagonal * (1.1 + staleFactor * 0.25),
        detectionBox.diagonal * (1.1 + staleFactor * 0.25)
      );
      const centroidDistance = distance(trackCenter, detectionCenter);

      if (centroidDistance > maxDistance) {
        return Number.POSITIVE_INFINITY;
      }

      let score = centroidDistance;

      if (track.rawId === detection.id) score -= 1000;
      if (track.personId != null && detection.personId != null && track.personId === detection.personId) {
        score -= 140;
      } else if (track.personId != null && detection.personId != null) {
        score += 120;
      }
      if (!pointsCompatible(stablePoints, detection.points)) score += 60;
      if (trackArea > 0 && detectionArea > 0) {
        const ratio = Math.max(trackArea, detectionArea) / Math.max(1, Math.min(trackArea, detectionArea));
        if (ratio > behavior.maxAreaRatio) score += 80;
      }
      if (track.missCount > 0) {
        score += track.missCount * 22 + missingForMs * 0.015;
      }

      return score;
    }

    function updateTrack(track, detection, nowMs) {
      const behavior = behaviorFor(partBehavior, detection.part);
      const stablePoints = track.lastStablePoints || track.renderPoints || track.lastSeenPoints || detection.points;
      const deltaMs = Math.max(16, nowMs - (track.lastSeenAt || nowMs));
      const stableDiff = shapeDifferenceScore(stablePoints, detection.points);
      const candidateBase = track.candidatePoints || stablePoints;
      const candidateDiff = shapeDifferenceScore(candidateBase, detection.points);
      const confirmFramesNeeded =
        stableDiff > 0.95 || (track.personId != null && detection.personId != null && track.personId !== detection.personId)
          ? behavior.strongConfirmFrames
          : behavior.confirmFrames;

      track.rawId = detection.id;
      track.personId = detection.personId;
      track.part = detection.part;
      track.lastSeenPoints = clonePoints(detection.points);
      track.lastSeenAt = nowMs;
      track.lastTouchedAt = nowMs;
      track.missingSince = null;
      track.seenCount += 1;
      track.targetAlpha = 1;
      track.missCount = 0;

      if (!Array.isArray(track.lastStablePoints) || track.lastStablePoints.length < 3) {
        track.lastStablePoints = clonePoints(detection.points);
        track.candidatePoints = clonePoints(detection.points);
        track.confirmCount = 1;
        track.velocity = zeroVelocity(detection.points);
        track.targetPoints = clonePoints(detection.points);
        if (!Array.isArray(track.renderPoints) || track.renderPoints.length < 3) {
          track.renderPoints = clonePoints(detection.points);
        }
        return;
      }

      if (stableDiff <= behavior.stableAcceptScore) {
        const mergedStable = smoothPolygon(track.lastStablePoints, detection.points, behavior.smooth);
        track.velocity = computeVelocity(track.lastStablePoints, mergedStable, track.velocity, deltaMs);
        track.lastStablePoints = clonePoints(mergedStable);
        track.candidatePoints = clonePoints(mergedStable);
        track.targetPoints = clonePoints(mergedStable);
        track.confirmCount = Math.max(1, confirmFramesNeeded);
      } else {
        if (candidateDiff <= behavior.candidateAcceptScore) {
          track.candidatePoints = smoothPolygon(candidateBase, detection.points, behavior.candidateBlend);
          track.confirmCount = (track.confirmCount || 0) + 1;
        } else {
          track.candidatePoints = clonePoints(detection.points);
          track.confirmCount = 1;
        }

        track.targetPoints = clonePoints(track.lastStablePoints);
        if (track.confirmCount >= confirmFramesNeeded) {
          const promoted = smoothPolygon(track.lastStablePoints, track.candidatePoints, behavior.promoteBlend);
          track.velocity = computeVelocity(track.lastStablePoints, promoted, track.velocity, deltaMs);
          track.lastStablePoints = clonePoints(promoted);
          track.targetPoints = clonePoints(promoted);
          track.candidatePoints = clonePoints(promoted);
          track.confirmCount = 0;
        }
      }

      if (!Array.isArray(track.renderPoints) || track.renderPoints.length < 3) {
        track.renderPoints = clonePoints(track.lastStablePoints);
      }
    }

    function createTrack(detection, nowMs) {
      const trackId = `stable_${nextTrackId++}`;
      tracks.set(trackId, {
        trackId,
        rawId: detection.id,
        personId: detection.personId,
        part: detection.part,
        renderPoints: clonePoints(detection.points),
        targetPoints: clonePoints(detection.points),
        lastStablePoints: clonePoints(detection.points),
        candidatePoints: clonePoints(detection.points),
        lastSeenPoints: clonePoints(detection.points),
        velocity: zeroVelocity(detection.points),
        fadeAlpha: 0,
        targetAlpha: 1,
        seenCount: 1,
        confirmCount: 1,
        missCount: 0,
        createdAt: nowMs,
        lastSeenAt: nowMs,
        lastTouchedAt: nowMs,
        lastRenderAt: nowMs,
        missingSince: null
      });
    }

    function ingest(rawPolygons, nowMs, enabledSettings) {
      const detections = [];
      for (const rawPoly of rawPolygons || []) {
        const poly = normalizeDetection(rawPoly);
        if (!poly) continue;
        if (enabledSettings && enabledSettings[poly.part] === false) continue;
        detections.push(poly);
      }
      const normalizedDetections = mergeEyesDetections(detections);

      const unmatchedTrackIds = new Set(tracks.keys());

      for (const detection of normalizedDetections) {
        let matchedTrackId = null;

        for (const trackId of unmatchedTrackIds) {
          const track = tracks.get(trackId);
          if (!track) continue;
          if (track.rawId === detection.id && track.part === detection.part) {
            matchedTrackId = trackId;
            break;
          }
        }

        if (matchedTrackId == null) {
          let bestTrackId = null;
          let bestScore = Number.POSITIVE_INFINITY;

          for (const trackId of unmatchedTrackIds) {
            const track = tracks.get(trackId);
            if (!track) continue;

            const score = matchScore(track, detection);
            if (score < bestScore) {
              bestScore = score;
              bestTrackId = trackId;
            }
          }

          if (bestTrackId != null && Number.isFinite(bestScore)) {
            matchedTrackId = bestTrackId;
          }
        }

        if (matchedTrackId == null) {
          createTrack(detection, nowMs);
          continue;
        }

        const track = tracks.get(matchedTrackId);
        if (!track) {
          createTrack(detection, nowMs);
          continue;
        }

        unmatchedTrackIds.delete(matchedTrackId);
        updateTrack(track, detection, nowMs);
      }

      for (const trackId of unmatchedTrackIds) {
        const track = tracks.get(trackId);
        if (!track) continue;
        if (track.missingSince == null) {
          track.missingSince = nowMs;
        }
        track.missCount = (track.missCount || 0) + 1;
      }
    }

    function advance(nowMs, enabledSettings, options = {}) {
      const output = [];
      const forceFadeAll = Boolean(options.forceFadeAll);

      for (const [trackId, track] of tracks.entries()) {
        const behavior = behaviorFor(partBehavior, track.part);
        const enabled = !enabledSettings || enabledSettings[track.part] !== false;
        const timeSinceSeen = Math.max(0, nowMs - (track.lastSeenAt || nowMs));
        const holdMs = Math.max(0, behavior.holdMs || 0);
        const fadeInMs = Math.max(1, behavior.fadeInMs || 1);
        const fadeOutMs = Math.max(1, behavior.fadeOutMs || 1);

        if (!enabled || forceFadeAll) {
          track.targetAlpha = 0;
        } else if (timeSinceSeen <= holdMs) {
          track.targetAlpha = 1;
          const stablePoints = track.lastStablePoints || track.targetPoints || track.lastSeenPoints;
          const predictMs = Math.min(timeSinceSeen, behavior.maxPredictMs || holdMs);
          track.targetPoints = predictPolygon(stablePoints, track.velocity, behavior.predict, predictMs);
        } else {
          track.targetAlpha = 0;
          track.targetPoints = clonePoints(track.lastStablePoints || track.targetPoints || track.renderPoints);
        }

        const deltaMs = Math.max(16, nowMs - (track.lastRenderAt || nowMs));
        const pointAlpha = 1 - Math.pow(1 - clamp01(behavior.smooth), deltaMs / 16.6667);
        track.renderPoints = lerpPoints(track.renderPoints || track.targetPoints, track.targetPoints, pointAlpha);

        const fadeMs = track.targetAlpha >= (track.fadeAlpha || 0) ? fadeInMs : fadeOutMs;
        const alphaStep = clamp01(deltaMs / fadeMs);
        track.fadeAlpha = (track.fadeAlpha || 0) + (track.targetAlpha - (track.fadeAlpha || 0)) * alphaStep;
        track.lastRenderAt = nowMs;

        const expired =
          track.targetAlpha <= 0 &&
          (track.fadeAlpha || 0) < 0.02 &&
          timeSinceSeen > holdMs + fadeOutMs * 1.5;

        if (expired) {
          tracks.delete(trackId);
          continue;
        }

        if ((track.fadeAlpha || 0) <= 0.02 || !Array.isArray(track.renderPoints) || track.renderPoints.length < 3) {
          continue;
        }

        output.push({
          id: track.rawId,
          stableId: track.trackId,
          part: track.part,
          person_id: track.personId,
          alpha: track.fadeAlpha || 0,
          points: clonePoints(track.renderPoints)
        });
      }

      return output;
    }

    return {
      clear,
      ingest,
      advance
    };
  }

  return {
    DEFAULT_BEHAVIOR,
    createLegacyOverlayState,
    createStableOverlayState
  };
});
