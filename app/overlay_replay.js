const fs = require("fs");
const path = require("path");

const {
  createLegacyOverlayState,
  createStableOverlayState
} = require("./overlay_state.js");

const PART_BEHAVIOR = {
  face: { smooth: 0.72, predict: 0.00, holdMs: 260, fadeInMs: 80, fadeOutMs: 150, stableAcceptScore: 0.44, candidateAcceptScore: 0.56 },
  eyes: { smooth: 0.84, predict: 0.00, holdMs: 260, fadeInMs: 80, fadeOutMs: 150, matchPx: 140, confirmFrames: 2, strongConfirmFrames: 3, candidateBlend: 0.44, promoteBlend: 0.22, stableAcceptScore: 0.28, candidateAcceptScore: 0.44, maxAreaRatio: 2.1 },
  shoulders: { smooth: 0.52, predict: 0.02, holdMs: 210, fadeInMs: 90, fadeOutMs: 135, stableAcceptScore: 0.48 },
  torso: { smooth: 0.42, predict: 0.07, holdMs: 380, fadeInMs: 110, fadeOutMs: 240, stableAcceptScore: 0.44, candidateAcceptScore: 0.54 },
  chest: { smooth: 0.36, predict: 0.05, holdMs: 280, fadeInMs: 100, fadeOutMs: 180 },
  back: { smooth: 0.36, predict: 0.05, holdMs: 300, fadeInMs: 100, fadeOutMs: 180 },
  armpits: { smooth: 0.48, predict: 0.02, holdMs: 220, fadeInMs: 90, fadeOutMs: 140, stableAcceptScore: 0.42, candidateAcceptScore: 0.52 },
  navel: { smooth: 0.44, predict: 0.02, holdMs: 220, fadeInMs: 90, fadeOutMs: 150 },
  hips: { smooth: 0.44, predict: 0.04, holdMs: 240, fadeInMs: 100, fadeOutMs: 170 },
  buttocks: { smooth: 0.44, predict: 0.03, holdMs: 320, fadeInMs: 100, fadeOutMs: 200 },
  hands: { smooth: 0.54, predict: 0.08, holdMs: 240, fadeInMs: 90, fadeOutMs: 150, stableAcceptScore: 0.46, candidateAcceptScore: 0.56 },
  wrists: { smooth: 0.58, predict: 0.00, holdMs: 180, fadeInMs: 80, fadeOutMs: 130 },
  forearms: { smooth: 0.47, predict: 0.08, holdMs: 210, fadeInMs: 90, fadeOutMs: 150 },
  thighs: { smooth: 0.50, predict: 0.08, holdMs: 240, fadeInMs: 100, fadeOutMs: 170 },
  knees: { smooth: 0.58, predict: 0.02, holdMs: 200, fadeInMs: 90, fadeOutMs: 140 },
  calves: { smooth: 0.48, predict: 0.06, holdMs: 230, fadeInMs: 100, fadeOutMs: 160 },
  feet: { smooth: 0.62, predict: 0.02, holdMs: 320, fadeInMs: 100, fadeOutMs: 170, confirmFrames: 2, strongConfirmFrames: 3, stableAcceptScore: 0.42, candidateAcceptScore: 0.54 },
  male_groin: { smooth: 0.46, predict: 0.00, holdMs: 360, fadeInMs: 100, fadeOutMs: 240, stableAcceptScore: 0.42, candidateAcceptScore: 0.52 },
  intimate_front: { smooth: 0.46, predict: 0.00, holdMs: 360, fadeInMs: 100, fadeOutMs: 240, stableAcceptScore: 0.42, candidateAcceptScore: 0.52 },
  intimate_back: { smooth: 0.44, predict: 0.00, holdMs: 320, fadeInMs: 100, fadeOutMs: 220 },
  silhouette: { smooth: 0.38, predict: 0.03, holdMs: 420, fadeInMs: 120, fadeOutMs: 300, confirmFrames: 2, stableAcceptScore: 0.46, candidateAcceptScore: 0.56 }
};

function buildEnabledSettings(partIds) {
  const out = {};
  for (const partId of partIds || []) {
    out[partId] = true;
  }
  return out;
}

function visibleParts(polygons, minAlpha = 0.35) {
  const parts = new Set();

  for (const polygon of polygons || []) {
    if (!polygon || !polygon.part) continue;
    if ((polygon.alpha ?? 1) < minAlpha) continue;
    if (!Array.isArray(polygon.points) || polygon.points.length < 3) continue;
    parts.add(String(polygon.part));
  }

  return parts;
}

function visiblePartCounts(polygons, minAlpha = 0.35) {
  const counts = {};

  for (const polygon of polygons || []) {
    if (!polygon || !polygon.part) continue;
    if ((polygon.alpha ?? 1) < minAlpha) continue;
    if (!Array.isArray(polygon.points) || polygon.points.length < 3) continue;
    const part = String(polygon.part);
    counts[part] = (counts[part] || 0) + 1;
  }

  return counts;
}

function countToggles(series) {
  let toggles = 0;
  for (let i = 1; i < series.length; i++) {
    if (series[i] !== series[i - 1]) {
      toggles += 1;
    }
  }
  return toggles;
}

function countMidstreamToggles(series) {
  const firstVisible = series.indexOf(true);
  const lastVisible = series.lastIndexOf(true);

  if (firstVisible < 0 || lastVisible <= firstVisible) {
    return 0;
  }

  let toggles = 0;
  for (let i = firstVisible + 1; i <= lastVisible; i++) {
    if (series[i] !== series[i - 1]) {
      toggles += 1;
    }
  }
  return toggles;
}

function countShortGaps(series, maxGapLength = 3) {
  let gaps = 0;
  let i = 0;

  while (i < series.length) {
    if (series[i]) {
      i += 1;
      continue;
    }

    const start = i;
    while (i < series.length && !series[i]) {
      i += 1;
    }
    const end = i - 1;
    const length = end - start + 1;
    const leftVisible = start > 0 && series[start - 1];
    const rightVisible = i < series.length && series[i];

    if (leftVisible && rightVisible && length <= maxGapLength) {
      gaps += 1;
    }
  }

  return gaps;
}

function countMidstreamMisses(series) {
  const firstVisible = series.indexOf(true);
  const lastVisible = series.lastIndexOf(true);

  if (firstVisible < 0 || lastVisible <= firstVisible) {
    return 0;
  }

  let misses = 0;
  for (let i = firstVisible; i <= lastVisible; i++) {
    if (!series[i]) {
      misses += 1;
    }
  }
  return misses;
}

function countSegmentMidstreamMisses(requiredSeries, visibleSeries) {
  let misses = 0;
  let index = 0;

  while (index < requiredSeries.length) {
    if (!requiredSeries[index]) {
      index += 1;
      continue;
    }

    const start = index;
    while (index < requiredSeries.length && requiredSeries[index]) {
      index += 1;
    }
    const end = index - 1;
    const segment = visibleSeries.slice(start, end + 1);
    misses += countMidstreamMisses(segment);
  }

  return misses;
}

function countFalseReappearances(requiredSeries, detectionSeries, visibleSeries) {
  let reappearances = 0;

  for (let i = 1; i < visibleSeries.length; i++) {
    if (!visibleSeries[i] || visibleSeries[i - 1]) {
      continue;
    }

    const hasDetection = detectionSeries[i];
    const isRequired = requiredSeries[i];
    const wasRequiredRecently = requiredSeries[Math.max(0, i - 1)];

    if (!hasDetection && !isRequired && !wasRequiredRecently) {
      reappearances += 1;
    }
  }

  return reappearances;
}

function holdAfterLossStats(detectionSeries, visibleSeries, frameDurationMs) {
  let totalFrames = 0;
  let maxFrames = 0;
  let events = 0;
  let index = 1;

  while (index < detectionSeries.length) {
    if (!(detectionSeries[index - 1] && !detectionSeries[index])) {
      index += 1;
      continue;
    }

    let frames = 0;
    let cursor = index;
    while (cursor < detectionSeries.length && !detectionSeries[cursor] && visibleSeries[cursor]) {
      frames += 1;
      cursor += 1;
    }

    if (frames > 0) {
      events += 1;
      totalFrames += frames;
      maxFrames = Math.max(maxFrames, frames);
    }

    index = cursor;
  }

  return {
    events,
    total_frames: totalFrames,
    total_ms: totalFrames * frameDurationMs,
    max_frames: maxFrames,
    max_ms: maxFrames * frameDurationMs
  };
}

function summarizeRenderedVideo(video, renderedFrames) {
  const allParts = new Set();
  const requiredParts = new Set();
  const frameDurationMs =
    video.frames.length > 1
      ? Math.max(1, (video.frames[video.frames.length - 1].time_ms - video.frames[0].time_ms) / (video.frames.length - 1))
      : 1000 / Math.max(1, video.fps || 6);

  for (const frame of video.frames) {
    for (const part of frame.required_parts || []) requiredParts.add(part);
    for (const part of frame.expected_parts || []) allParts.add(part);
    for (const part of frame.detected_parts || []) allParts.add(part);
  }

  for (const frame of renderedFrames) {
    for (const part of frame.visible_parts) allParts.add(part);
  }

  const perPart = {};
  let totalToggles = 0;
  let totalMidstreamToggles = 0;
  let totalShortGaps = 0;
  let totalHoldAfterLossFrames = 0;
  let totalHoldAfterLossMs = 0;
  let totalHoldAfterLossEvents = 0;
  let totalFalseReappearances = 0;
  let eyeFragmentedFrames = 0;
  let maxHoldAfterLossFrames = 0;
  let maxHoldAfterLossMs = 0;

  for (const frame of renderedFrames) {
    if ((frame.visible_part_counts?.eyes || 0) > 1) {
      eyeFragmentedFrames += 1;
    }
  }

  for (const part of Array.from(allParts).sort()) {
    const series = renderedFrames.map((frame) => frame.visible_parts.has(part));
    const requiredSeries = renderedFrames.map((frame) => frame.required_parts.includes(part));
    const detectionSeries = video.frames.map((frame) => (frame.detected_parts || []).includes(part));
    const toggles = countToggles(series);
    const midstreamToggles = countMidstreamToggles(series);
    const shortGaps = countShortGaps(series);
    const visibleFrames = series.filter(Boolean).length;
    const holdStats = holdAfterLossStats(detectionSeries, series, frameDurationMs);
    const falseReappearances = countFalseReappearances(requiredSeries, detectionSeries, series);

    perPart[part] = {
      toggles,
      midstream_toggles: midstreamToggles,
      short_gaps: shortGaps,
      visible_frames: visibleFrames,
      hold_after_loss_events: holdStats.events,
      hold_after_loss_frames: holdStats.total_frames,
      hold_after_loss_ms: holdStats.total_ms,
      max_hold_after_loss_frames: holdStats.max_frames,
      max_hold_after_loss_ms: holdStats.max_ms,
      false_reappearances: falseReappearances
    };

    totalToggles += toggles;
    totalMidstreamToggles += midstreamToggles;
    totalShortGaps += shortGaps;
    totalHoldAfterLossFrames += holdStats.total_frames;
    totalHoldAfterLossMs += holdStats.total_ms;
    totalHoldAfterLossEvents += holdStats.events;
    totalFalseReappearances += falseReappearances;
    maxHoldAfterLossFrames = Math.max(maxHoldAfterLossFrames, holdStats.max_frames);
    maxHoldAfterLossMs = Math.max(maxHoldAfterLossMs, holdStats.max_ms);
  }

  let requiredMissFrames = 0;
  let midstreamRequiredMissFrames = 0;
  const requiredMissByPart = {};
  const midstreamRequiredMissByPart = {};

  for (const part of Array.from(requiredParts).sort()) {
    const requiredSeries = renderedFrames.map((frame) => frame.required_parts.includes(part));
    const visibleSeries = renderedFrames.map((frame) => frame.visible_parts.has(part));
    const midstreamMisses = countSegmentMidstreamMisses(requiredSeries, visibleSeries);
    if (midstreamMisses > 0) {
      midstreamRequiredMissByPart[part] = midstreamMisses;
      midstreamRequiredMissFrames += midstreamMisses;
    }
  }

  for (const frame of renderedFrames) {
    for (const part of frame.required_parts) {
      if (!frame.visible_parts.has(part)) {
        requiredMissFrames += 1;
        requiredMissByPart[part] = (requiredMissByPart[part] || 0) + 1;
      }
    }
  }

  return {
    frame_count: renderedFrames.length,
    total_toggles: totalToggles,
    total_midstream_toggles: totalMidstreamToggles,
    total_short_gaps: totalShortGaps,
    required_miss_frames: requiredMissFrames,
    midstream_required_miss_frames: midstreamRequiredMissFrames,
    hold_after_loss_events: totalHoldAfterLossEvents,
    hold_after_loss_frames: totalHoldAfterLossFrames,
    hold_after_loss_ms: totalHoldAfterLossMs,
    max_hold_after_loss_frames: maxHoldAfterLossFrames,
    max_hold_after_loss_ms: maxHoldAfterLossMs,
    false_reappearances: totalFalseReappearances,
    eye_fragmented_frames: eyeFragmentedFrames,
    required_miss_by_part: requiredMissByPart,
    midstream_required_miss_by_part: midstreamRequiredMissByPart,
    per_part: perPart
  };
}

function replayVideo(video, stateFactory, enabledSettings) {
  const state = stateFactory();
  const renderedFrames = [];

  for (const frame of video.frames) {
    state.ingest(frame.polygons || [], frame.time_ms, enabledSettings);
    const rendered = state.advance(frame.time_ms, enabledSettings, { forceFadeAll: false });
    renderedFrames.push({
      frame_index: frame.frame_index,
      required_parts: frame.required_parts || [],
      visible_parts: visibleParts(rendered),
      visible_part_counts: visiblePartCounts(rendered)
    });
  }

  return summarizeRenderedVideo(video, renderedFrames);
}

function compareSummaries(legacy, stable) {
  const perPart = {};
  const parts = new Set([
    ...Object.keys(legacy.per_part || {}),
    ...Object.keys(stable.per_part || {})
  ]);

  for (const part of Array.from(parts).sort()) {
    const oldStats = legacy.per_part?.[part] || { toggles: 0, short_gaps: 0, visible_frames: 0 };
    const newStats = stable.per_part?.[part] || { toggles: 0, short_gaps: 0, visible_frames: 0 };
    perPart[part] = {
      toggles: {
        legacy: oldStats.toggles,
        stable: newStats.toggles,
        delta: newStats.toggles - oldStats.toggles
      },
      short_gaps: {
        legacy: oldStats.short_gaps,
        stable: newStats.short_gaps,
        delta: newStats.short_gaps - oldStats.short_gaps
      },
      visible_frames: {
        legacy: oldStats.visible_frames,
        stable: newStats.visible_frames,
        delta: newStats.visible_frames - oldStats.visible_frames
      },
      hold_after_loss_frames: {
        legacy: oldStats.hold_after_loss_frames || 0,
        stable: newStats.hold_after_loss_frames || 0,
        delta: (newStats.hold_after_loss_frames || 0) - (oldStats.hold_after_loss_frames || 0)
      },
      max_hold_after_loss_frames: {
        legacy: oldStats.max_hold_after_loss_frames || 0,
        stable: newStats.max_hold_after_loss_frames || 0,
        delta: (newStats.max_hold_after_loss_frames || 0) - (oldStats.max_hold_after_loss_frames || 0)
      },
      false_reappearances: {
        legacy: oldStats.false_reappearances || 0,
        stable: newStats.false_reappearances || 0,
        delta: (newStats.false_reappearances || 0) - (oldStats.false_reappearances || 0)
      }
    };
  }

  return {
    total_toggles: {
      legacy: legacy.total_toggles,
      stable: stable.total_toggles,
      delta: stable.total_toggles - legacy.total_toggles
    },
    total_midstream_toggles: {
      legacy: legacy.total_midstream_toggles,
      stable: stable.total_midstream_toggles,
      delta: stable.total_midstream_toggles - legacy.total_midstream_toggles
    },
    total_short_gaps: {
      legacy: legacy.total_short_gaps,
      stable: stable.total_short_gaps,
      delta: stable.total_short_gaps - legacy.total_short_gaps
    },
    required_miss_frames: {
      legacy: legacy.required_miss_frames,
      stable: stable.required_miss_frames,
      delta: stable.required_miss_frames - legacy.required_miss_frames
    },
    midstream_required_miss_frames: {
      legacy: legacy.midstream_required_miss_frames,
      stable: stable.midstream_required_miss_frames,
      delta: stable.midstream_required_miss_frames - legacy.midstream_required_miss_frames
    },
    hold_after_loss_frames: {
      legacy: legacy.hold_after_loss_frames,
      stable: stable.hold_after_loss_frames,
      delta: stable.hold_after_loss_frames - legacy.hold_after_loss_frames
    },
    hold_after_loss_ms: {
      legacy: legacy.hold_after_loss_ms,
      stable: stable.hold_after_loss_ms,
      delta: stable.hold_after_loss_ms - legacy.hold_after_loss_ms
    },
    max_hold_after_loss_frames: {
      legacy: legacy.max_hold_after_loss_frames,
      stable: stable.max_hold_after_loss_frames,
      delta: stable.max_hold_after_loss_frames - legacy.max_hold_after_loss_frames
    },
    max_hold_after_loss_ms: {
      legacy: legacy.max_hold_after_loss_ms,
      stable: stable.max_hold_after_loss_ms,
      delta: stable.max_hold_after_loss_ms - legacy.max_hold_after_loss_ms
    },
    false_reappearances: {
      legacy: legacy.false_reappearances,
      stable: stable.false_reappearances,
      delta: stable.false_reappearances - legacy.false_reappearances
    },
    eye_fragmented_frames: {
      legacy: legacy.eye_fragmented_frames || 0,
      stable: stable.eye_fragmented_frames || 0,
      delta: (stable.eye_fragmented_frames || 0) - (legacy.eye_fragmented_frames || 0)
    },
    required_miss_by_part: {
      legacy: legacy.required_miss_by_part,
      stable: stable.required_miss_by_part
    },
    midstream_required_miss_by_part: {
      legacy: legacy.midstream_required_miss_by_part,
      stable: stable.midstream_required_miss_by_part
    },
    per_part: perPart
  };
}

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];

  if (!inputPath || !outputPath) {
    console.error("Usage: node app/overlay_replay.js <trace.json> <comparison.json>");
    process.exit(1);
  }

  const trace = JSON.parse(fs.readFileSync(inputPath, "utf-8"));
  const enabledSettings = buildEnabledSettings(trace.part_ids || []);

  const result = {
    created_at: new Date().toISOString(),
    source_trace: path.resolve(inputPath),
    videos: [],
    overall: null
  };

  let legacyToggleTotal = 0;
  let stableToggleTotal = 0;
  let legacyMidstreamToggleTotal = 0;
  let stableMidstreamToggleTotal = 0;
  let legacyGapTotal = 0;
  let stableGapTotal = 0;
  let legacyRequiredMissTotal = 0;
  let stableRequiredMissTotal = 0;
  let legacyMidstreamRequiredMissTotal = 0;
  let stableMidstreamRequiredMissTotal = 0;
  let legacyHoldAfterLossFramesTotal = 0;
  let stableHoldAfterLossFramesTotal = 0;
  let legacyHoldAfterLossMsTotal = 0;
  let stableHoldAfterLossMsTotal = 0;
  let legacyMaxHoldAfterLossFrames = 0;
  let stableMaxHoldAfterLossFrames = 0;
  let legacyMaxHoldAfterLossMs = 0;
  let stableMaxHoldAfterLossMs = 0;
  let legacyFalseReappearancesTotal = 0;
  let stableFalseReappearancesTotal = 0;
  let legacyEyeFragmentedFrames = 0;
  let stableEyeFragmentedFrames = 0;

  for (const video of trace.videos || []) {
    const legacy = replayVideo(
      video,
      () => createLegacyOverlayState({ partBehavior: PART_BEHAVIOR, defaultHoldFrames: 6 }),
      enabledSettings
    );
    const stable = replayVideo(
      video,
      () => createStableOverlayState({ partBehavior: PART_BEHAVIOR }),
      enabledSettings
    );
    const comparison = compareSummaries(legacy, stable);

    legacyToggleTotal += legacy.total_toggles;
    stableToggleTotal += stable.total_toggles;
    legacyMidstreamToggleTotal += legacy.total_midstream_toggles;
    stableMidstreamToggleTotal += stable.total_midstream_toggles;
    legacyGapTotal += legacy.total_short_gaps;
    stableGapTotal += stable.total_short_gaps;
    legacyRequiredMissTotal += legacy.required_miss_frames;
    stableRequiredMissTotal += stable.required_miss_frames;
    legacyMidstreamRequiredMissTotal += legacy.midstream_required_miss_frames;
    stableMidstreamRequiredMissTotal += stable.midstream_required_miss_frames;
    legacyHoldAfterLossFramesTotal += legacy.hold_after_loss_frames;
    stableHoldAfterLossFramesTotal += stable.hold_after_loss_frames;
    legacyHoldAfterLossMsTotal += legacy.hold_after_loss_ms;
    stableHoldAfterLossMsTotal += stable.hold_after_loss_ms;
    legacyMaxHoldAfterLossFrames = Math.max(legacyMaxHoldAfterLossFrames, legacy.max_hold_after_loss_frames);
    stableMaxHoldAfterLossFrames = Math.max(stableMaxHoldAfterLossFrames, stable.max_hold_after_loss_frames);
    legacyMaxHoldAfterLossMs = Math.max(legacyMaxHoldAfterLossMs, legacy.max_hold_after_loss_ms);
    stableMaxHoldAfterLossMs = Math.max(stableMaxHoldAfterLossMs, stable.max_hold_after_loss_ms);
    legacyFalseReappearancesTotal += legacy.false_reappearances;
    stableFalseReappearancesTotal += stable.false_reappearances;
    legacyEyeFragmentedFrames += legacy.eye_fragmented_frames || 0;
    stableEyeFragmentedFrames += stable.eye_fragmented_frames || 0;

    result.videos.push({
      video_name: video.video_name,
      legacy,
      stable,
      comparison
    });
  }

  result.overall = {
    toggles: {
      legacy: legacyToggleTotal,
      stable: stableToggleTotal,
      delta: stableToggleTotal - legacyToggleTotal
    },
    midstream_toggles: {
      legacy: legacyMidstreamToggleTotal,
      stable: stableMidstreamToggleTotal,
      delta: stableMidstreamToggleTotal - legacyMidstreamToggleTotal
    },
    short_gaps: {
      legacy: legacyGapTotal,
      stable: stableGapTotal,
      delta: stableGapTotal - legacyGapTotal
    },
    required_miss_frames: {
      legacy: legacyRequiredMissTotal,
      stable: stableRequiredMissTotal,
      delta: stableRequiredMissTotal - legacyRequiredMissTotal
    },
    midstream_required_miss_frames: {
      legacy: legacyMidstreamRequiredMissTotal,
      stable: stableMidstreamRequiredMissTotal,
      delta: stableMidstreamRequiredMissTotal - legacyMidstreamRequiredMissTotal
    },
    hold_after_loss_frames: {
      legacy: legacyHoldAfterLossFramesTotal,
      stable: stableHoldAfterLossFramesTotal,
      delta: stableHoldAfterLossFramesTotal - legacyHoldAfterLossFramesTotal
    },
    hold_after_loss_ms: {
      legacy: legacyHoldAfterLossMsTotal,
      stable: stableHoldAfterLossMsTotal,
      delta: stableHoldAfterLossMsTotal - legacyHoldAfterLossMsTotal
    },
    max_hold_after_loss_frames: {
      legacy: legacyMaxHoldAfterLossFrames,
      stable: stableMaxHoldAfterLossFrames,
      delta: stableMaxHoldAfterLossFrames - legacyMaxHoldAfterLossFrames
    },
    max_hold_after_loss_ms: {
      legacy: legacyMaxHoldAfterLossMs,
      stable: stableMaxHoldAfterLossMs,
      delta: stableMaxHoldAfterLossMs - legacyMaxHoldAfterLossMs
    },
    false_reappearances: {
      legacy: legacyFalseReappearancesTotal,
      stable: stableFalseReappearancesTotal,
      delta: stableFalseReappearancesTotal - legacyFalseReappearancesTotal
    },
    eye_fragmented_frames: {
      legacy: legacyEyeFragmentedFrames,
      stable: stableEyeFragmentedFrames,
      delta: stableEyeFragmentedFrames - legacyEyeFragmentedFrames
    }
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), "utf-8");
  console.log(JSON.stringify(result.overall, null, 2));
}

main();
