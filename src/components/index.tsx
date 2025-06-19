/* eslint-disable react-hooks/exhaustive-deps */
import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import * as tf from "@tensorflow/tfjs";
import {
  Radar,
  Target,
  Play,
  Pause,
  RotateCcw,
  AlertTriangle,
  Eye,
  Brain,
  Activity,
  TrendingUp,
} from "lucide-react";
interface AIModels {
  threatClassifier: tf.LayersModel | null;
  trajectoryPredictor: tf.LayersModel | null;
  decisionMaker: tf.LayersModel | null;
}

interface TrainingData {
  threatFeatures: number[][];
  threatLabels: number[][];
  trajectoryData: number[][];
  trajectoryTargets: number[][];
  decisionFeatures: number[][];
  decisionTargets: number[][];
}
interface Threat {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  detected: boolean;
  tracked: boolean;
  intercepted: boolean;
  destroyed: boolean;
  age: number;
  type: string;
  priority?: number;
  distanceToCenter?: number;
  detectedAt?: number;
  confidence?: number;
  threatLevel?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  predictionAccuracy?: number;
  lastPosition?: { x: number; y: number };
  behaviorPattern?: string;
  estimatedImpactTime?: number;
  destroyedAt?: number;
  missedTarget?: boolean;
  aiClassification?: {
    confidence: number;
    probabilities: { [key: string]: number };
    anomalyScore: number;
  };
  aiTrajectory?: {
    predictedPoints: { x: number; y: number; t: number }[];
    accuracy: number;
  };
}

interface Alert {
  id: number;
  type:
    | "THREAT_DETECTED"
    | "HIGH_PRIORITY"
    | "INTERCEPTION"
    | "SYSTEM"
    | "TARGET_DESTROYED"
    | "TARGET_MISSED"
    | "AI_ALERT";
  message: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  timestamp: number;
  duration: number;
}

interface Interceptor {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  target: number;
  targetX: number;
  targetY: number;
  launched: boolean;
  successProbability?: number;
  fuel?: number;
  status?: "READY" | "LAUNCHED" | "TRACKING" | "IMPACT" | "DESTROYED";
  hasHitTarget?: boolean;
  aiDecision?: {
    confidence: number;
    reasoning: string;
    optimalLaunchTime: number;
  };
}

interface TrajectoryPoint {
  x: number;
  y: number;
  t: number;
  confidence?: number;
}

interface InterceptionPoint {
  x: number;
  y: number;
  t: number;
  probability?: number;
}

interface ThreatConfig {
  speed: number;
  size: number;
  color: string;
  priority: number;
  detectability: number;
}

interface SystemMetrics {
  threatsDetected: number;
  successfulInterceptions: number;
  missedTargets: number;
  falsePositives: number;
  systemUptime: number;
  averageResponseTime: number;
  destroyedThreats: number;
  escapedThreats: number;
  aiAccuracy: number;
  modelTrainingCount: number;
}

interface Explosion {
  id: number;
  x: number;
  y: number;
  radius: number;
  age: number;
  maxAge: number;
  success: boolean;
}

const AirDefenseSimulation = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const lastUpdateTime = useRef<number>(0);
  const systemStartTime = useRef<number>(Date.now());

  const [isRunning, setIsRunning] = useState(false);
  const [detectedThreats, setDetectedThreats] = useState<Threat[]>([]);
  const [interceptors, setInterceptors] = useState<Interceptor[]>([]);
  const [radarSweep, setRadarSweep] = useState(0);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [explosions, setExplosions] = useState<Explosion[]>([]);
  const [aiModels, setAiModels] = useState<AIModels>({
    threatClassifier: null,
    trajectoryPredictor: null,
    decisionMaker: null,
  });
  const [trainingData, setTrainingData] = useState<TrainingData>({
    threatFeatures: [],
    threatLabels: [],
    trajectoryData: [],
    trajectoryTargets: [],
    decisionFeatures: [],
    decisionTargets: [],
  });
  const [aiEnabled, setAiEnabled] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    threatsDetected: 0,
    successfulInterceptions: 0,
    missedTargets: 0,
    falsePositives: 0,
    systemUptime: 0,
    averageResponseTime: 0,
    destroyedThreats: 0,
    escapedThreats: 0,
    aiAccuracy: 0,
    modelTrainingCount: 0,
  });
  const [radarSensitivity, setRadarSensitivity] = useState(0.8);
  const [defenseMode, setDefenseMode] = useState<
    "DEFENSIVE" | "AGGRESSIVE" | "ADAPTIVE" | "AI_AUTONOMOUS"
  >("AI_AUTONOMOUS");

  const addAlert = useCallback(
    (type: Alert["type"], message: string, severity: Alert["severity"]) => {
      const newAlert: Alert = {
        id: Date.now() + Math.random(),
        type,
        message,
        severity,
        timestamp: Date.now(),
        duration:
          severity === "CRITICAL" ? 8000 : severity === "HIGH" ? 5000 : 3000,
      };

      setAlerts((prev) => {
        const updated = [newAlert, ...prev.slice(0, MAX_ALERTS - 1)];
        return updated;
      });
    },
    []
  );

  const CANVAS_WIDTH = 800;
  const CANVAS_HEIGHT = 600;
  const RADAR_CENTER_X = CANVAS_WIDTH / 2;
  const RADAR_CENTER_Y = CANVAS_HEIGHT / 2;
  const RADAR_RADIUS = 250;
  const THREAT_SPAWN_RATE = 0.015;
  const MAX_THREATS = 10;
  const TARGET_FPS = 60;
  const FRAME_TIME = 1000 / TARGET_FPS;
  const MAX_ALERTS = 5;
  const INTERCEPTION_DISTANCE = 15;

  const THREAT_TYPES: Record<string, ThreatConfig> = {
    DRONE: {
      speed: 2,
      size: 4,
      color: "#ff6b6b",
      priority: 1,
      detectability: 0.9,
    },
    MISSILE: {
      speed: 5,
      size: 6,
      color: "#ff3838",
      priority: 4,
      detectability: 0.7,
    },
    AIRCRAFT: {
      speed: 3,
      size: 8,
      color: "#ffa500",
      priority: 2,
      detectability: 0.95,
    },
    STEALTH: {
      speed: 4,
      size: 5,
      color: "#666",
      priority: 3,
      detectability: 0.4,
    },
    SWARM: {
      speed: 2.5,
      size: 3,
      color: "#ff9999",
      priority: 2,
      detectability: 0.8,
    },
    UNKNOWN: {
      speed: 2.5,
      size: 5,
      color: "#999",
      priority: 1,
      detectability: 0.6,
    },
  };

  const initializeAIModels = useCallback(async () => {
    try {
      const threatClassifier = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [8], units: 32, activation: "relu" }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 16, activation: "relu" }),
          tf.layers.dropout({ rate: 0.1 }),
          tf.layers.dense({ units: 6, activation: "softmax" }),
        ],
      });

      threatClassifier.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      const trajectoryPredictor = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [10], units: 64, activation: "relu" }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 32, activation: "relu" }),
          tf.layers.dense({ units: 16, activation: "relu" }),
          tf.layers.dense({ units: 4, activation: "linear" }),
        ],
      });

      trajectoryPredictor.compile({
        optimizer: tf.train.adam(0.001),
        loss: "meanSquaredError",
        metrics: ["mae"],
      });

      const decisionMaker = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [12], units: 48, activation: "relu" }),
          tf.layers.dropout({ rate: 0.3 }),
          tf.layers.dense({ units: 24, activation: "relu" }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 12, activation: "relu" }),
          tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
      });

      decisionMaker.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      setAiModels({
        threatClassifier,
        trajectoryPredictor,
        decisionMaker,
      });

      addAlert(
        "AI_ALERT",
        "AI models initialized successfully - Neural networks online",
        "MEDIUM"
      );
    } catch (error) {
      console.error("Failed to initialize AI models:", error);
      addAlert(
        "AI_ALERT",
        "AI model initialization failed - Falling back to traditional methods",
        "HIGH"
      );
    }
  }, []);

  useEffect(() => {
    initializeAIModels();
  }, [initializeAIModels]);

  const extractThreatFeatures = useCallback((threat: Threat): number[] => {
    const distanceFromCenter = Math.sqrt(
      Math.pow(threat.x - RADAR_CENTER_X, 2) +
        Math.pow(threat.y - RADAR_CENTER_Y, 2)
    );
    const speed = Math.sqrt(threat.vx * threat.vx + threat.vy * threat.vy);
    const direction = Math.atan2(threat.vy, threat.vx);
    const approachAngle = Math.atan2(
      RADAR_CENTER_Y - threat.y,
      RADAR_CENTER_X - threat.x
    );

    return [
      threat.size / 10,
      speed / 6,
      distanceFromCenter / RADAR_RADIUS,
      direction / (2 * Math.PI),
      approachAngle / (2 * Math.PI),
      threat.age / 100,
      threat.confidence || 0.5,
      Math.abs(direction - approachAngle) / Math.PI,
    ];
  }, []);
  const classifyThreatWithAI = useCallback(
    async (
      threat: Threat
    ): Promise<{
      type: string;
      confidence: number;
      threatLevel: Threat["threatLevel"];
      aiClassification: Threat["aiClassification"];
    }> => {
      if (!aiModels.threatClassifier || !aiEnabled) {
        const speed = Math.sqrt(threat.vx * threat.vx + threat.vy * threat.vy);
        const size = threat.size;

        let type = "UNKNOWN";
        let confidence = 0.5;
        let threatLevel: Threat["threatLevel"] = "LOW";

        if (speed > 4.5 && size > 5) {
          type = "MISSILE";
          confidence = 0.9;
          threatLevel = "CRITICAL";
        } else if (speed > 3.5 && size < 5) {
          type = "STEALTH";
          confidence = 0.6;
          threatLevel = "HIGH";
        } else if (speed < 2.5 && size < 5) {
          type = Math.random() < 0.3 ? "SWARM" : "DRONE";
          confidence = 0.8;
          threatLevel = type === "SWARM" ? "MEDIUM" : "LOW";
        } else if (size > 7) {
          type = "AIRCRAFT";
          confidence = 0.85;
          threatLevel = "MEDIUM";
        }

        return {
          type,
          confidence,
          threatLevel,
          aiClassification: {
            confidence: confidence,
            probabilities: { [type]: confidence },
            anomalyScore: 0.1,
          },
        };
      }

      try {
        const features = extractThreatFeatures(threat);
        const prediction = aiModels.threatClassifier.predict(
          tf.tensor2d([features])
        ) as tf.Tensor;
        const probabilities = await prediction.data();
        prediction.dispose();

        const threatTypeLabels = [
          "DRONE",
          "MISSILE",
          "AIRCRAFT",
          "STEALTH",
          "SWARM",
          "UNKNOWN",
        ];
        const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
        const type = threatTypeLabels[maxProbIndex];
        const confidence = probabilities[maxProbIndex];
        const entropy = -Array.from(probabilities).reduce(
          (sum: number, p: number) => sum + (p > 0 ? p * Math.log(p) : 0),
          0
        );
        const anomalyScore = entropy / Math.log(threatTypeLabels.length);

        let threatLevel: Threat["threatLevel"] = "LOW";
        if (type === "MISSILE" || confidence > 0.8) threatLevel = "CRITICAL";
        else if (type === "STEALTH" || confidence > 0.6) threatLevel = "HIGH";
        else if (type === "AIRCRAFT" || type === "SWARM")
          threatLevel = "MEDIUM";

        const aiClassification = {
          confidence,
          probabilities: Object.fromEntries(
            threatTypeLabels.map((label, i) => [label, probabilities[i]])
          ),
          anomalyScore,
        };
        if (anomalyScore > 0.8) {
          addAlert(
            "AI_ALERT",
            `Unusual threat pattern detected! Anomaly score: ${Math.round(
              anomalyScore * 100
            )}%`,
            "HIGH"
          );
        }

        return { type, confidence, threatLevel, aiClassification };
      } catch (error) {
        console.error("AI classification error:", error);

        return classifyThreatWithAI(threat);
      }
    },
    [aiModels.threatClassifier, aiEnabled, extractThreatFeatures, addAlert]
  );
  const predictTrajectoryWithAI = useCallback(
    async (threat: Threat, steps = 15): Promise<TrajectoryPoint[]> => {
      if (!aiModels.trajectoryPredictor || !aiEnabled) {
        const trajectory: TrajectoryPoint[] = [];
        let x = threat.x;
        let y = threat.y;
        let vx = threat.vx;
        let vy = threat.vy;

        for (let i = 0; i < steps; i++) {
          x += vx;
          y += vy;
          vx += (Math.random() - 0.5) * 0.1;
          vy += (Math.random() - 0.5) * 0.1;
          trajectory.push({ x, y, t: i, confidence: 0.5 });
        }
        return trajectory;
      }

      try {
        const trajectory: TrajectoryPoint[] = [];
        let currentX = threat.x;
        let currentY = threat.y;
        let currentVx = threat.vx;
        let currentVy = threat.vy;

        for (let i = 0; i < steps; i++) {
          const features = [
            currentX / CANVAS_WIDTH,
            currentY / CANVAS_HEIGHT,
            currentVx / 6,
            currentVy / 6,
            threat.size / 10,
            i / steps,
            threat.confidence || 0.5,
            Math.sqrt(currentVx * currentVx + currentVy * currentVy) / 6,
            Math.atan2(currentVy, currentVx) / (2 * Math.PI),
            Math.sqrt(
              Math.pow(currentX - RADAR_CENTER_X, 2) +
                Math.pow(currentY - RADAR_CENTER_Y, 2)
            ) / RADAR_RADIUS,
          ];

          const prediction = aiModels.trajectoryPredictor.predict(
            tf.tensor2d([features])
          ) as tf.Tensor;
          const result = await prediction.data();
          prediction.dispose();

          currentX = result[0] * CANVAS_WIDTH;
          currentY = result[1] * CANVAS_HEIGHT;
          currentVx = result[2] * 6;
          currentVy = result[3] * 6;

          const confidence = Math.max(0.1, 0.9 - i * 0.04);
          trajectory.push({ x: currentX, y: currentY, t: i, confidence });
        }

        return trajectory;
      } catch (error) {
        console.error("AI trajectory prediction error:", error);
        return predictTrajectoryWithAI(threat, steps);
      }
    },
    [aiModels.trajectoryPredictor, aiEnabled]
  );
  const makeInterceptionDecisionWithAI = useCallback(
    async (
      threat: Threat,
      interceptor: Partial<Interceptor>
    ): Promise<{
      decision: "LAUNCH" | "WAIT" | "ABORT";
      confidence: number;
      reasoning: string;
    }> => {
      if (!aiModels.decisionMaker || !aiEnabled) {
        return {
          decision: "LAUNCH",
          confidence: 0.7,
          reasoning: "Traditional algorithm decision",
        };
      }

      try {
        const distanceToThreat = Math.sqrt(
          Math.pow(threat.x - RADAR_CENTER_X, 2) +
            Math.pow(threat.y - RADAR_CENTER_Y, 2)
        );
        const threatSpeed = Math.sqrt(
          threat.vx * threat.vx + threat.vy * threat.vy
        );
        const timeToImpact = distanceToThreat / threatSpeed;

        const features = [
          threat.size / 10,
          threatSpeed / 6,
          distanceToThreat / RADAR_RADIUS,
          threat.confidence || 0.5,
          (threat.priority || 1) / 4,
          timeToImpact / 100,
          interceptor.successProbability || 0.5,
          (interceptor.fuel || 100) / 100,
          interceptors.length / 5,
          (threat.age || 0) / 100,
          radarSensitivity,
          defenseMode === "AGGRESSIVE"
            ? 1
            : defenseMode === "DEFENSIVE"
            ? 0
            : 0.5,
        ];

        const prediction = aiModels.decisionMaker.predict(
          tf.tensor2d([features])
        ) as tf.Tensor;
        const probabilities = await prediction.data();
        prediction.dispose();

        const decisions = ["LAUNCH", "WAIT", "ABORT"] as const;
        const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
        const decision = decisions[maxProbIndex];
        const confidence = probabilities[maxProbIndex];

        let reasoning = "";
        if (decision === "LAUNCH") {
          reasoning = `High threat priority (${threat.threatLevel}), optimal interception window`;
        } else if (decision === "WAIT") {
          reasoning = `Monitoring threat movement, waiting for better opportunity`;
        } else {
          reasoning = `Low success probability or resource conservation needed`;
        }

        return { decision, confidence, reasoning };
      } catch (error) {
        console.error("AI decision making error:", error);
        return {
          decision: "LAUNCH",
          confidence: 0.5,
          reasoning: "AI error - fallback decision",
        };
      }
    },
    [
      aiModels.decisionMaker,
      aiEnabled,
      interceptors.length,
      radarSensitivity,
      defenseMode,
    ]
  );
  const collectTrainingData = useCallback(
    (threat: Threat, interceptor: Interceptor, success: boolean) => {
      if (!aiEnabled) return;
      const threatFeatures = extractThreatFeatures(threat);
      const threatTypeLabels = [
        "DRONE",
        "MISSILE",
        "AIRCRAFT",
        "STEALTH",
        "SWARM",
        "UNKNOWN",
      ];
      const threatLabel = new Array(6).fill(0);
      const typeIndex = threatTypeLabels.indexOf(threat.type);
      if (typeIndex !== -1) {
        threatLabel[typeIndex] = 1;
      }
      const decisionFeatures = [
        threat.size / 10,
        Math.sqrt(threat.vx * threat.vx + threat.vy * threat.vy) / 6,
        Math.sqrt(
          Math.pow(threat.x - RADAR_CENTER_X, 2) +
            Math.pow(threat.y - RADAR_CENTER_Y, 2)
        ) / RADAR_RADIUS,
        threat.confidence || 0.5,
        (threat.priority || 1) / 4,
        threat.age / 100,
        interceptor.successProbability || 0.5,
        (interceptor.fuel || 100) / 100,
        interceptors.length / 5,
        threat.age / 100,
        radarSensitivity,
        defenseMode === "AGGRESSIVE"
          ? 1
          : defenseMode === "DEFENSIVE"
          ? 0
          : 0.5,
      ];

      const decisionLabel = success ? [1, 0, 0] : [0, 0, 1];

      setTrainingData((prev) => ({
        ...prev,
        threatFeatures: [...prev.threatFeatures, threatFeatures],
        threatLabels: [...prev.threatLabels, threatLabel],
        decisionFeatures: [...prev.decisionFeatures, decisionFeatures],
        decisionTargets: [...prev.decisionTargets, decisionLabel],
      }));
    },
    [
      aiEnabled,
      extractThreatFeatures,
      interceptors.length,
      radarSensitivity,
      defenseMode,
    ]
  );
  const trainAIModels = useCallback(async () => {
    if (!aiEnabled || trainingData.threatFeatures.length < 10) return;

    try {
      if (aiModels.threatClassifier && trainingData.threatFeatures.length > 0) {
        const xs = tf.tensor2d(trainingData.threatFeatures);
        const ys = tf.tensor2d(trainingData.threatLabels);

        await aiModels.threatClassifier.fit(xs, ys, {
          epochs: 5,
          batchSize: 8,
          validationSplit: 0.2,
          verbose: 0,
        });

        xs.dispose();
        ys.dispose();
      }
      if (aiModels.decisionMaker && trainingData.decisionFeatures.length > 0) {
        const xs = tf.tensor2d(trainingData.decisionFeatures);
        const ys = tf.tensor2d(trainingData.decisionTargets);

        await aiModels.decisionMaker.fit(xs, ys, {
          epochs: 3,
          batchSize: 4,
          validationSplit: 0.2,
          verbose: 0,
        });

        xs.dispose();
        ys.dispose();
      }

      setSystemMetrics((prev) => ({
        ...prev,
        modelTrainingCount: prev.modelTrainingCount + 1,
        aiAccuracy: Math.min(0.95, prev.aiAccuracy + 0.05),
      }));

      addAlert(
        "AI_ALERT",
        `AI models retrained! Training cycles: ${
          systemMetrics.modelTrainingCount + 1
        }`,
        "MEDIUM"
      );
      setTrainingData({
        threatFeatures: [],
        threatLabels: [],
        trajectoryData: [],
        trajectoryTargets: [],
        decisionFeatures: [],
        decisionTargets: [],
      });
    } catch (error) {
      console.error("AI training error:", error);
      addAlert(
        "AI_ALERT",
        "AI model training failed - Models may need reinitialization",
        "HIGH"
      );
    }
  }, [
    aiEnabled,
    trainingData,
    aiModels,
    systemMetrics.modelTrainingCount,
    addAlert,
  ]);
  const classifyThreat = useCallback(
    (
      threat: Threat
    ): {
      type: string;
      confidence: number;
      threatLevel: Threat["threatLevel"];
    } => {
      const speed = Math.sqrt(threat.vx * threat.vx + threat.vy * threat.vy);
      const size = threat.size;
      const distanceFromCenter = Math.sqrt(
        Math.pow(threat.x - RADAR_CENTER_X, 2) +
          Math.pow(threat.y - RADAR_CENTER_Y, 2)
      );

      let type = "UNKNOWN";
      let confidence = 0.5;
      let threatLevel: Threat["threatLevel"] = "LOW";
      if (speed > 4.5 && size > 5) {
        type = "MISSILE";
        confidence = 0.9;
        threatLevel = "CRITICAL";
      } else if (speed > 3.5 && size < 5 && Math.random() < 0.3) {
        type = "STEALTH";
        confidence = 0.6;
        threatLevel = "HIGH";
      } else if (speed < 2.5 && size < 5) {
        type = Math.random() < 0.3 ? "SWARM" : "DRONE";
        confidence = 0.8;
        threatLevel = type === "SWARM" ? "MEDIUM" : "LOW";
      } else if (size > 7) {
        type = "AIRCRAFT";
        confidence = 0.85;
        threatLevel = "MEDIUM";
      }
      confidence *=
        (1 - distanceFromCenter / (RADAR_RADIUS * 1.5)) * radarSensitivity;
      confidence = Math.max(0.1, Math.min(0.99, confidence));

      return { type, confidence, threatLevel };
    },
    [radarSensitivity]
  );
  const predictTrajectory = useCallback(
    (threat: Threat, steps = 15): TrajectoryPoint[] => {
      const trajectory: TrajectoryPoint[] = [];
      let x = threat.x;
      let y = threat.y;
      let vx = threat.vx;
      let vy = threat.vy;
      const basePredictionAccuracy =
        THREAT_TYPES[threat.type]?.detectability || 0.5;
      const ageBonus = Math.min(threat.age / 50, 0.3);
      const predictionAccuracy = Math.min(
        basePredictionAccuracy + ageBonus,
        0.95
      );

      for (let i = 0; i < steps; i++) {
        const adaptiveFactor = 1 + 0.1 * Math.sin(i * 0.2);

        x += vx * adaptiveFactor;
        y += vy * adaptiveFactor;
        const uncertainty = (1 - predictionAccuracy) * 2;
        vx += (Math.random() - 0.5) * uncertainty;
        vy += (Math.random() - 0.5) * uncertainty;
        if (threat.type === "MISSILE") {
          vy += 0.05;
          vx *= 0.999;
          vy *= 0.999;
        }

        trajectory.push({
          x,
          y,
          t: i,
          confidence: predictionAccuracy * (1 - i * 0.05),
        });
      }
      return trajectory;
    },
    []
  );
  const calculateInterceptionPoint = useCallback(
    (threat: Threat, interceptorSpeed = 6): InterceptionPoint | null => {
      const dx = threat.x - RADAR_CENTER_X;
      const dy = threat.y - RADAR_CENTER_Y;
      const threatSpeed = Math.sqrt(
        threat.vx * threat.vx + threat.vy * threat.vy
      );

      const a = threatSpeed * threatSpeed - interceptorSpeed * interceptorSpeed;
      const b = 2 * (dx * threat.vx + dy * threat.vy);
      const c = dx * dx + dy * dy;

      const discriminant = b * b - 4 * a * c;
      if (discriminant < 0) return null;

      const t = (-b - Math.sqrt(discriminant)) / (2 * a);
      if (t < 0) return null;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const timeUntilImpact = t;
      const threatPriority = THREAT_TYPES[threat.type]?.priority || 1;
      const detectionConfidence = threat.confidence || 0.5;

      let probability = 0.8;
      probability *= detectionConfidence;
      probability *= Math.max(0.3, 1 - distance / RADAR_RADIUS);
      probability *= Math.max(0.4, 1 - timeUntilImpact / 100);
      probability *= threatPriority / 4;
      if (defenseMode === "AGGRESSIVE") probability *= 1.2;
      else if (defenseMode === "DEFENSIVE") probability *= 0.9;

      return {
        x: threat.x + threat.vx * t,
        y: threat.y + threat.vy * t,
        t: t,
        probability: Math.min(0.95, Math.max(0.1, probability)),
      };
    },
    [defenseMode]
  );
  const prioritizeThreats = useCallback((threats: Threat[]): Threat[] => {
    return threats
      .map((threat) => {
        const distance = Math.sqrt(
          Math.pow(threat.x - RADAR_CENTER_X, 2) +
            Math.pow(threat.y - RADAR_CENTER_Y, 2)
        );
        const speed = Math.sqrt(threat.vx * threat.vx + threat.vy * threat.vy);
        const estimatedImpactTime = distance / speed;
        let threatScore = 0;
        threatScore += (THREAT_TYPES[threat.type]?.priority || 1) * 25;
        threatScore += (1 - distance / RADAR_RADIUS) * 30;
        threatScore += Math.min(speed / 6, 1) * 20;
        threatScore += (threat.confidence || 0.5) * 15;
        threatScore += Math.max(0, 100 - estimatedImpactTime) * 0.5;
        const levelMultiplier = {
          LOW: 1,
          MEDIUM: 1.5,
          HIGH: 2,
          CRITICAL: 3,
        };
        threatScore *= levelMultiplier[threat.threatLevel || "LOW"];

        return {
          ...threat,
          priority: Math.round(threatScore),
          estimatedImpactTime,
        };
      })
      .sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }, []);
  const createThreat = useCallback((): Threat => {
    const patterns = ["random", "coordinated", "stealth_approach", "swarm"];
    const pattern = patterns[Math.floor(Math.random() * patterns.length)];

    let angle, spawnDistance, speed, targetAngle;

    switch (pattern) {
      case "coordinated": {
        angle = Math.PI / 4 + (Math.random() * Math.PI) / 2;
        spawnDistance = RADAR_RADIUS + 50 + Math.random() * 50;
        speed = 2 + Math.random() * 3;
        targetAngle = Math.atan2(
          RADAR_CENTER_Y - (RADAR_CENTER_Y + Math.sin(angle) * spawnDistance),
          RADAR_CENTER_X - (RADAR_CENTER_X + Math.cos(angle) * spawnDistance)
        );
        break;
      }

      case "stealth_approach": {
        angle = Math.random() * 2 * Math.PI;
        spawnDistance = RADAR_RADIUS + 100;
        speed = 1.5 + Math.random() * 2;
        targetAngle = Math.random() * 2 * Math.PI;
        break;
      }

      case "swarm": {
        const swarmCenter = Math.random() * 2 * Math.PI;
        angle = swarmCenter + (Math.random() - 0.5) * 0.5;
        spawnDistance = RADAR_RADIUS + 30;
        speed = 2 + Math.random() * 2;
        targetAngle = Math.random() * 2 * Math.PI;
        break;
      }

      default: {
        angle = Math.random() * 2 * Math.PI;
        spawnDistance = RADAR_RADIUS + 50;
        speed = 1 + Math.random() * 4;
        targetAngle = Math.random() * 2 * Math.PI;
      }
    }

    const threat: Threat = {
      id: Date.now() + Math.random(),
      x: RADAR_CENTER_X + Math.cos(angle) * spawnDistance,
      y: RADAR_CENTER_Y + Math.sin(angle) * spawnDistance,
      vx: Math.cos(targetAngle) * speed,
      vy: Math.sin(targetAngle) * speed,
      size: 3 + Math.random() * 6,
      detected: false,
      tracked: false,
      intercepted: false,
      destroyed: false,
      age: 0,
      type: "UNKNOWN",
      behaviorPattern: pattern,
    };
    if (aiEnabled && aiModels.threatClassifier) {
      const classification = classifyThreat(threat);
      threat.type = classification.type;
      threat.confidence = classification.confidence;
      threat.threatLevel = classification.threatLevel;
    } else {
      const classification = classifyThreat(threat);
      threat.type = classification.type;
      threat.confidence = classification.confidence;
      threat.threatLevel = classification.threatLevel;
    }

    return threat;
  }, [classifyThreat]);
  const launchInterceptor = useCallback(
    (threat: Threat): Interceptor | null => {
      const interceptionPoint = calculateInterceptionPoint(threat);
      if (!interceptionPoint || interceptionPoint.probability! < 0.3)
        return null;

      const dx = interceptionPoint.x - RADAR_CENTER_X;
      const dy = interceptionPoint.y - RADAR_CENTER_Y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const speed = 6 + (threat.threatLevel === "CRITICAL" ? 2 : 0);

      return {
        id: Date.now() + Math.random(),
        x: RADAR_CENTER_X,
        y: RADAR_CENTER_Y,
        vx: (dx / distance) * speed,
        vy: (dy / distance) * speed,
        target: threat.id,
        targetX: interceptionPoint.x,
        targetY: interceptionPoint.y,
        launched: true,
        successProbability: interceptionPoint.probability,
        fuel: 100,
        status: "LAUNCHED",
      };
    },
    [calculateInterceptionPoint]
  );
  const detectThreats = useCallback(
    (threats: Threat[], radarAngle: number): Threat[] => {
      const detectionRange = RADAR_RADIUS * radarSensitivity;
      const detectionAngle = Math.PI / 8;

      return threats.map((threat) => {
        const dx = threat.x - RADAR_CENTER_X;
        const dy = threat.y - RADAR_CENTER_Y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);

        let angleDiff = Math.abs(angle - radarAngle);
        if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;

        const inRange = distance <= detectionRange;
        const inAngle = angleDiff <= detectionAngle;
        const threatConfig = THREAT_TYPES[threat.type] || THREAT_TYPES.UNKNOWN;
        let detectionProbability = 0.1;

        if (inRange && inAngle) {
          detectionProbability = threatConfig.detectability * radarSensitivity;
          detectionProbability *= 1 - (distance / detectionRange) * 0.5;
          detectionProbability *= Math.min(1.5, threat.size / 5);
          detectionProbability *= 0.8 + Math.random() * 0.4;
        }

        if (!threat.detected && Math.random() < detectionProbability) {
          threat.detected = true;
          threat.detectedAt = Date.now();
          if (
            threat.threatLevel === "CRITICAL" ||
            threat.threatLevel === "HIGH"
          ) {
            addAlert(
              "THREAT_DETECTED",
              `${threat.threatLevel} priority ${threat.type} detected`,
              threat.threatLevel
            );
          }

          setSystemMetrics((prev) => ({
            ...prev,
            threatsDetected: prev.threatsDetected + 1,
          }));
        }

        return threat;
      });
    },
    [radarSensitivity, addAlert]
  );
  useEffect(() => {
    const interval = setInterval(() => {
      setAlerts((prev) =>
        prev.filter((alert) => Date.now() - alert.timestamp < alert.duration)
      );
    }, 1000);

    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics((prev) => ({
        ...prev,
        systemUptime: Date.now() - systemStartTime.current,
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    if (!aiEnabled) return;

    const trainingInterval = setInterval(() => {
      if (trainingData.threatFeatures.length >= 20) {
        trainAIModels();
      }
    }, 30000);

    return () => clearInterval(trainingInterval);
  }, [aiEnabled, trainingData.threatFeatures.length, trainAIModels]);
  const createExplosion = useCallback(
    (x: number, y: number, success: boolean) => {
      const explosion: Explosion = {
        id: Date.now() + Math.random(),
        x,
        y,
        radius: 0,
        age: 0,
        maxAge: success ? 30 : 20,
        success,
      };
      setExplosions((prev) => [...prev, explosion]);
    },
    []
  );
  const checkInterceptions = useCallback(() => {
    setInterceptors((prevInterceptors) => {
      const updatedInterceptors = [...prevInterceptors];

      setDetectedThreats((prevThreats) => {
        const updatedThreats = [...prevThreats];

        updatedInterceptors.forEach((interceptor, interceptorIndex) => {
          if (interceptor.status === "IMPACT" || interceptor.hasHitTarget)
            return;

          const threatIndex = updatedThreats.findIndex(
            (threat) => threat.id === interceptor.target
          );
          if (threatIndex === -1) return;

          const threat = updatedThreats[threatIndex];
          if (threat.destroyed) return;
          const distance = Math.sqrt(
            Math.pow(interceptor.x - threat.x, 2) +
              Math.pow(interceptor.y - threat.y, 2)
          );

          if (distance < INTERCEPTION_DISTANCE) {
            const successChance = interceptor.successProbability || 0.5;
            const isSuccess = Math.random() < successChance;

            if (isSuccess) {
              updatedThreats[threatIndex] = {
                ...threat,
                destroyed: true,
                destroyedAt: Date.now(),
              };
              updatedInterceptors[interceptorIndex] = {
                ...interceptor,
                status: "IMPACT",
                hasHitTarget: true,
              };
              createExplosion(threat.x, threat.y, true);
              addAlert(
                "TARGET_DESTROYED",
                `${threat.type} destroyed! Hit probability: ${Math.round(
                  successChance * 100
                )}%`,
                "MEDIUM"
              );
              collectTrainingData(threat, interceptor, true);
              setSystemMetrics((prev) => ({
                ...prev,
                successfulInterceptions: prev.successfulInterceptions + 1,
                destroyedThreats: prev.destroyedThreats + 1,
              }));
            } else {
              updatedInterceptors[interceptorIndex] = {
                ...interceptor,
                status: "DESTROYED",
              };
              createExplosion(interceptor.x, interceptor.y, false);
              addAlert(
                "TARGET_MISSED",
                `Miss! ${threat.type} evaded interception`,
                "HIGH"
              );
              collectTrainingData(threat, interceptor, false);
              setSystemMetrics((prev) => ({
                ...prev,
                missedTargets: prev.missedTargets + 1,
              }));
            }
          }
        });

        return updatedThreats;
      });

      return updatedInterceptors;
    });
  }, [createExplosion, addAlert]);
  const updateSimulation = useCallback(() => {
    const currentTime = performance.now();
    if (currentTime - lastUpdateTime.current < FRAME_TIME) {
      return;
    }
    lastUpdateTime.current = currentTime;

    setDetectedThreats((prevThreats) => {
      let threats = [...prevThreats];
      let spawnRate = THREAT_SPAWN_RATE;
      if (defenseMode === "AGGRESSIVE") spawnRate *= 1.5;
      else if (defenseMode === "DEFENSIVE") spawnRate *= 0.7;
      if (threats.length < MAX_THREATS && Math.random() < spawnRate) {
        threats.push(createThreat());
      }
      threats = threats.map((threat) => {
        if (threat.destroyed) return threat;

        const updatedThreat = {
          ...threat,
          x: threat.x + threat.vx,
          y: threat.y + threat.vy,
          age: threat.age + 1,
          lastPosition: { x: threat.x, y: threat.y },
        };
        if (threat.detected) {
          updatedThreat.predictionAccuracy = Math.min(
            0.95,
            (threat.predictionAccuracy || 0.5) + 0.01
          );
        }

        return updatedThreat;
      });
      threats = threats.map((threat) => {
        if (threat.destroyed || threat.missedTarget) return threat;

        const distanceFromCenter = Math.sqrt(
          Math.pow(threat.x - RADAR_CENTER_X, 2) +
            Math.pow(threat.y - RADAR_CENTER_Y, 2)
        );
        if (
          distanceFromCenter < 20 ||
          distanceFromCenter > RADAR_RADIUS + 200 ||
          threat.age > 600
        ) {
          if (!threat.intercepted && threat.detected) {
            setSystemMetrics((prev) => ({
              ...prev,
              escapedThreats: prev.escapedThreats + 1,
            }));

            if (distanceFromCenter < 20) {
              addAlert(
                "TARGET_MISSED",
                `${threat.type} reached target! Air defense system failed to stop it`,
                "CRITICAL"
              );
            }
          }

          return {
            ...threat,
            missedTarget: true,
          };
        }

        return threat;
      });
      threats = threats.filter((threat) => {
        if (
          threat.destroyed &&
          threat.destroyedAt &&
          Date.now() - threat.destroyedAt > 3000
        ) {
          return false;
        }
        if (threat.missedTarget && threat.age > 100) {
          return false;
        }
        return true;
      });
      threats = detectThreats(threats, radarSweep);

      return threats;
    });
    setInterceptors((prevInterceptors) => {
      return prevInterceptors
        .map((interceptor) => {
          if (
            interceptor.status === "IMPACT" ||
            interceptor.status === "DESTROYED"
          ) {
            return interceptor;
          }

          const updatedInterceptor = {
            ...interceptor,
            x: interceptor.x + interceptor.vx,
            y: interceptor.y + interceptor.vy,
            fuel: Math.max(0, (interceptor.fuel || 100) - 0.5),
          };
          const distanceToTarget = Math.sqrt(
            Math.pow(updatedInterceptor.x - updatedInterceptor.targetX, 2) +
              Math.pow(updatedInterceptor.y - updatedInterceptor.targetY, 2)
          );

          if (distanceToTarget < 50) {
            updatedInterceptor.status = "TRACKING";
          }

          return updatedInterceptor;
        })
        .filter((interceptor) => {
          if ((interceptor.fuel || 0) <= 0) return false;
          if (interceptor.status === "DESTROYED") return false;

          const distance = Math.sqrt(
            Math.pow(interceptor.x - RADAR_CENTER_X, 2) +
              Math.pow(interceptor.y - RADAR_CENTER_Y, 2)
          );
          return distance < RADAR_RADIUS + 150;
        });
    });
    setExplosions((prev) =>
      prev
        .map((explosion) => ({
          ...explosion,
          age: explosion.age + 1,
          radius: Math.min(explosion.age * 2, explosion.success ? 25 : 15),
        }))
        .filter((explosion) => explosion.age < explosion.maxAge)
    );
    checkInterceptions();
    const threatDensity = detectedThreats.filter((t) => t.detected).length;
    const adaptiveSpeed = 0.03 + threatDensity * 0.01;
    setRadarSweep((prev) => (prev + adaptiveSpeed) % (2 * Math.PI));
  }, [
    createThreat,
    detectThreats,
    radarSweep,
    defenseMode,
    detectedThreats,
    checkInterceptions,
  ]);
  const threatStats = useMemo(
    () => ({
      total: detectedThreats.length,
      detected: detectedThreats.filter((t) => t.detected).length,
      interceptors: interceptors.length,
      critical: detectedThreats.filter((t) => t.threatLevel === "CRITICAL")
        .length,
      high: detectedThreats.filter((t) => t.threatLevel === "HIGH").length,
      medium: detectedThreats.filter((t) => t.threatLevel === "MEDIUM").length,
      destroyed: detectedThreats.filter((t) => t.destroyed).length,
      escaped: detectedThreats.filter((t) => t.missedTarget).length,
      avgConfidence:
        detectedThreats
          .filter((t) => t.detected)
          .reduce((acc, t) => acc + (t.confidence || 0), 0) /
        Math.max(1, detectedThreats.filter((t) => t.detected).length),
    }),
    [detectedThreats, interceptors]
  );

  const prioritizedThreats = useMemo(() => {
    return prioritizeThreats(detectedThreats.filter((t) => t.detected)).slice(
      0,
      6
    );
  }, [detectedThreats, prioritizeThreats]);
  useEffect(() => {
    if (!isRunning) return;

    const detectedAndTracked = detectedThreats.filter(
      (t) => t.detected && !t.intercepted
    );
    const prioritized = prioritizeThreats(detectedAndTracked);
    const maxInterceptors =
      defenseMode === "AGGRESSIVE" ? 4 : defenseMode === "DEFENSIVE" ? 2 : 3;
    const activeInterceptors = interceptors.filter(
      (i) => i.status !== "IMPACT"
    ).length;

    prioritized
      .slice(0, maxInterceptors - activeInterceptors)
      .forEach((threat) => {
        const hasInterceptor = interceptors.some((i) => i.target === threat.id);
        let launchProbability = 0.3;
        if (threat.threatLevel === "CRITICAL") launchProbability = 0.8;
        else if (threat.threatLevel === "HIGH") launchProbability = 0.6;
        else if (threat.threatLevel === "MEDIUM") launchProbability = 0.4;

        if (defenseMode === "AGGRESSIVE") launchProbability *= 1.5;

        if (!hasInterceptor && Math.random() < launchProbability) {
          const interceptor = launchInterceptor(threat);
          if (interceptor) {
            if (defenseMode === "AI_AUTONOMOUS" && aiEnabled) {
              makeInterceptionDecisionWithAI(threat, interceptor).then(
                (decision) => {
                  if (decision.decision === "LAUNCH") {
                    setInterceptors((prev) => [
                      ...prev,
                      {
                        ...interceptor,
                        aiDecision: {
                          confidence: decision.confidence,
                          reasoning: decision.reasoning,
                          optimalLaunchTime: Date.now(),
                        },
                      },
                    ]);
                    setDetectedThreats((prev) =>
                      prev.map((t) =>
                        t.id === threat.id ? { ...t, intercepted: true } : t
                      )
                    );

                    addAlert(
                      "INTERCEPTION",
                      `AI Interceptor launched: ${
                        decision.reasoning
                      } (${Math.round(decision.confidence * 100)}% confidence)`,
                      threat.threatLevel || "MEDIUM"
                    );
                  }
                }
              );
            } else {
              setInterceptors((prev) => [...prev, interceptor]);
              setDetectedThreats((prev) =>
                prev.map((t) =>
                  t.id === threat.id ? { ...t, intercepted: true } : t
                )
              );

              addAlert(
                "INTERCEPTION",
                `Interceptor launched against ${threat.type} (${Math.round(
                  (interceptor.successProbability || 0) * 100
                )}% success rate)`,
                threat.threatLevel || "MEDIUM"
              );
            }
          }
        }
      });
  }, [
    detectedThreats,
    interceptors,
    isRunning,
    prioritizeThreats,
    launchInterceptor,
    defenseMode,
    addAlert,
  ]);
  useEffect(() => {
    if (isRunning) {
      const animate = () => {
        updateSimulation();
        if (isRunning) {
          animationRef.current = requestAnimationFrame(animate);
        }
      };
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isRunning, updateSimulation]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    const gradient = ctx.createRadialGradient(
      RADAR_CENTER_X,
      RADAR_CENTER_Y,
      0,
      RADAR_CENTER_X,
      RADAR_CENTER_Y,
      RADAR_RADIUS
    );
    gradient.addColorStop(0, "rgba(0, 255, 0, 0.1)");
    gradient.addColorStop(1, "rgba(0, 255, 0, 0.02)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 1;
    for (let r = 50; r <= RADAR_RADIUS; r += 50) {
      ctx.beginPath();
      ctx.arc(RADAR_CENTER_X, RADAR_CENTER_Y, r, 0, 2 * Math.PI);
      ctx.globalAlpha = 0.3 + (r / RADAR_RADIUS) * 0.4;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const sweepIntensity = Math.min(1, 0.5 + threatStats.detected * 0.1);
    ctx.strokeStyle = `rgba(0, 255, 0, ${sweepIntensity * 0.8})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(RADAR_CENTER_X, RADAR_CENTER_Y);
    ctx.lineTo(
      RADAR_CENTER_X + Math.cos(radarSweep) * RADAR_RADIUS,
      RADAR_CENTER_Y + Math.sin(radarSweep) * RADAR_RADIUS
    );
    ctx.stroke();
    const coneColor =
      threatStats.critical > 0
        ? "rgba(255, 0, 0, 0.2)"
        : threatStats.high > 0
        ? "rgba(255, 165, 0, 0.2)"
        : "rgba(0, 255, 0, 0.1)";
    ctx.fillStyle = coneColor;
    ctx.beginPath();
    ctx.moveTo(RADAR_CENTER_X, RADAR_CENTER_Y);
    ctx.arc(
      RADAR_CENTER_X,
      RADAR_CENTER_Y,
      RADAR_RADIUS * radarSensitivity,
      radarSweep - Math.PI / 16,
      radarSweep + Math.PI / 16
    );
    ctx.closePath();
    ctx.fill();
    detectedThreats.forEach((threat) => {
      const threatConfig = THREAT_TYPES[threat.type] || THREAT_TYPES.UNKNOWN;

      if (threat.detected) {
        const trajectory = predictTrajectory(threat, 20);
        ctx.strokeStyle = threatConfig.color + "60";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(threat.x, threat.y);
        trajectory.forEach((point, index) => {
          const alpha = Math.max(
            0.1,
            (point.confidence || 0.5) * (1 - index * 0.03)
          );
          ctx.globalAlpha = alpha;
          ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
        const threatRadius =
          threat.size +
          (threat.threatLevel === "CRITICAL"
            ? 3
            : threat.threatLevel === "HIGH"
            ? 2
            : threat.threatLevel === "MEDIUM"
            ? 1
            : 0);
        const pulseEffect =
          threat.threatLevel === "CRITICAL" || threat.threatLevel === "HIGH"
            ? 1 + Math.sin(Date.now() * 0.01) * 0.3
            : 1;

        ctx.fillStyle = threatConfig.color;
        ctx.beginPath();
        ctx.arc(threat.x, threat.y, threatRadius * pulseEffect, 0, 2 * Math.PI);
        ctx.fill();
        if (threat.intercepted) {
          ctx.strokeStyle = "#00aaff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(threat.x, threat.y, threatRadius + 5, 0, 2 * Math.PI);
          ctx.stroke();
        }
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px monospace";
        ctx.fillText(threat.type, threat.x + 12, threat.y - 8);

        ctx.font = "8px monospace";
        ctx.fillStyle = "#ccc";
        ctx.fillText(
          `${Math.round((threat.confidence || 0) * 100)}%`,
          threat.x + 12,
          threat.y + 2
        );
        ctx.fillText(`P${threat.priority || 0}`, threat.x + 12, threat.y + 12);
        const levelColor = {
          CRITICAL: "#ff0000",
          HIGH: "#ff6600",
          MEDIUM: "#ffcc00",
          LOW: "#66ff66",
        }[threat.threatLevel || "LOW"];

        ctx.fillStyle = levelColor;
        ctx.fillRect(
          threat.x - threatRadius - 2,
          threat.y - threatRadius - 8,
          3,
          6
        );
      } else {
        ctx.fillStyle = "#333";
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.arc(threat.x, threat.y, threat.size, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    });
    interceptors.forEach((interceptor) => {
      const statusColors = {
        LAUNCHED: "#00aaff",
        TRACKING: "#00ff00",
        IMPACT: "#ff6600",
        READY: "#ffffff",
        DESTROYED: "#ff0000",
      };

      ctx.fillStyle = statusColors[interceptor.status || "READY"];
      ctx.beginPath();
      ctx.arc(interceptor.x, interceptor.y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = statusColors[interceptor.status || "READY"] + "80";
      ctx.lineWidth = interceptor.status === "TRACKING" ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(interceptor.x, interceptor.y);
      ctx.lineTo(
        interceptor.x - interceptor.vx * 8,
        interceptor.y - interceptor.vy * 8
      );
      ctx.stroke();
      if (
        interceptor.successProbability &&
        interceptor.successProbability > 0.5
      ) {
        ctx.fillStyle = "#fff";
        ctx.font = "8px monospace";
        ctx.fillText(
          `${Math.round(interceptor.successProbability * 100)}%`,
          interceptor.x + 6,
          interceptor.y - 6
        );
      }
      if ((interceptor.fuel || 100) < 30) {
        ctx.strokeStyle = "#ff6600";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(interceptor.x, interceptor.y, 6, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });
    explosions.forEach((explosion) => {
      const alpha = 1 - explosion.age / explosion.maxAge;
      const radius = explosion.radius;

      if (explosion.success) {
        const gradient = ctx.createRadialGradient(
          explosion.x,
          explosion.y,
          0,
          explosion.x,
          explosion.y,
          radius
        );
        gradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
        gradient.addColorStop(0.3, `rgba(0, 255, 0, ${alpha * 0.8})`);
        gradient.addColorStop(1, `rgba(0, 255, 0, ${alpha * 0.2})`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(explosion.x, explosion.y, radius, 0, 2 * Math.PI);
        ctx.fill();
        if (explosion.age < 15) {
          ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
          ctx.font = "bold 12px monospace";
          ctx.textAlign = "center";
          ctx.fillText("DESTROYED", explosion.x, explosion.y - radius - 5);
          ctx.textAlign = "left";
        }
      } else {
        const gradient = ctx.createRadialGradient(
          explosion.x,
          explosion.y,
          0,
          explosion.x,
          explosion.y,
          radius
        );
        gradient.addColorStop(0, `rgba(255, 100, 100, ${alpha})`);
        gradient.addColorStop(0.5, `rgba(255, 0, 0, ${alpha * 0.6})`);
        gradient.addColorStop(1, `rgba(255, 0, 0, ${alpha * 0.1})`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(explosion.x, explosion.y, radius, 0, 2 * Math.PI);
        ctx.fill();
        if (explosion.age < 10) {
          ctx.fillStyle = `rgba(255, 100, 100, ${alpha})`;
          ctx.font = "bold 10px monospace";
          ctx.textAlign = "center";
          ctx.fillText("MISS", explosion.x, explosion.y - radius - 3);
          ctx.textAlign = "left";
        }
      }
    });
    const centerColor = isRunning
      ? threatStats.critical > 0
        ? "#ff0000"
        : threatStats.high > 0
        ? "#ff6600"
        : "#00ff00"
      : "#666";

    ctx.fillStyle = centerColor;
    ctx.beginPath();
    ctx.arc(RADAR_CENTER_X, RADAR_CENTER_Y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = centerColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(RADAR_CENTER_X, RADAR_CENTER_Y, 10, 0, 2 * Math.PI);
    ctx.stroke();
  }, [
    detectedThreats,
    interceptors,
    radarSweep,
    predictTrajectory,
    threatStats,
    radarSensitivity,
    isRunning,
  ]);

  const resetSimulation = () => {
    setDetectedThreats([]);
    setInterceptors([]);
    setRadarSweep(0);
    setAlerts([]);
    setExplosions([]);
    setSystemMetrics({
      threatsDetected: 0,
      successfulInterceptions: 0,
      missedTargets: 0,
      falsePositives: 0,
      systemUptime: 0,
      averageResponseTime: 0,
      destroyedThreats: 0,
      escapedThreats: 0,
      aiAccuracy: 0,
      modelTrainingCount: 0,
    });
    systemStartTime.current = Date.now();
    addAlert("SYSTEM", "System reset complete", "LOW");
  };
  const formatUptime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    return `${hours.toString().padStart(2, "0")}:${(minutes % 60)
      .toString()
      .padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gray-900 text-green-400 font-mono">
      {/* Enhanced Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
          <Radar className="w-8 h-8" />
          Advanced AI Air Defense System
        </h1>
        <p className="text-gray-400">
          Neural threat detection, predictive trajectory analysis and autonomous
          interception
        </p>
      </div>

      {/* Real-time Alert System */}
      {alerts.length > 0 && (
        <div className="mb-4 space-y-2">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-3 rounded-lg border-l-4 flex items-center gap-3 animate-pulse ${
                alert.severity === "CRITICAL"
                  ? "bg-red-900/50 border-red-500 text-red-200"
                  : alert.severity === "HIGH"
                  ? "bg-orange-900/50 border-orange-500 text-orange-200"
                  : alert.severity === "MEDIUM"
                  ? "bg-yellow-900/50 border-yellow-500 text-yellow-200"
                  : "bg-blue-900/50 border-blue-500 text-blue-200"
              }`}
            >
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <div className="flex-1">
                <div className="flex justify-between items-center">
                  <span className="font-semibold">
                    {alert.type.replace("_", " ")}
                  </span>
                  <span className="text-xs opacity-75">
                    {Math.round((Date.now() - alert.timestamp) / 1000)}s ago
                  </span>
                </div>
                <p className="text-sm">{alert.message}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Enhanced Radar Display */}
        <div className="lg:col-span-3">
          <div className="bg-black border border-green-500 rounded-lg p-4">
            <div className="mb-4 flex justify-between items-center">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <Eye className="w-5 h-5" />
                Neural Radar Display
              </h2>
              <div className="flex gap-2">
                <select
                  value={defenseMode}
                  onChange={(e) =>
                    setDefenseMode(
                      e.target.value as
                        | "DEFENSIVE"
                        | "ADAPTIVE"
                        | "AGGRESSIVE"
                        | "AI_AUTONOMOUS"
                    )
                  }
                  className="px-3 py-1 bg-gray-800 border border-gray-600 rounded text-sm"
                >
                  <option value="DEFENSIVE">Defensive</option>
                  <option value="ADAPTIVE">Adaptive</option>
                  <option value="AGGRESSIVE">Aggressive</option>
                  <option value="AI_AUTONOMOUS">AI Autonomous</option>
                </select>
                <button
                  onClick={() => setAiEnabled(!aiEnabled)}
                  className={`px-3 py-1 rounded text-sm transition-colors ${
                    aiEnabled
                      ? "bg-green-600 hover:bg-green-700 text-white"
                      : "bg-gray-600 hover:bg-gray-700 text-gray-300"
                  }`}
                >
                  AI: {aiEnabled ? "ON" : "OFF"}
                </button>
                <button
                  onClick={trainAIModels}
                  disabled={
                    !aiEnabled || trainingData.threatFeatures.length < 5
                  }
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm transition-colors"
                >
                  Train ({trainingData.threatFeatures.length})
                </button>
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded transition-colors duration-200"
                >
                  {isRunning ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  {isRunning ? "Pause" : "Start"}
                </button>
                <button
                  onClick={resetSimulation}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors duration-200"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
              </div>
            </div>
            <div
              className="relative bg-black border border-green-500 rounded"
              style={{
                width: "100%",
                maxWidth: "800px",
                aspectRatio: "800/600",
                margin: "0 auto",
              }}
            >
              <canvas
                ref={canvasRef}
                width={CANVAS_WIDTH}
                height={CANVAS_HEIGHT}
                className="w-full h-full object-contain"
                style={{
                  display: "block",
                  imageRendering: "crisp-edges",
                }}
              />
            </div>

            {/* Radar Controls */}
            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div>
                <label className="block text-gray-400 mb-1">
                  Radar Sensitivity
                </label>
                <input
                  type="range"
                  min="0.3"
                  max="1.0"
                  step="0.1"
                  value={radarSensitivity}
                  onChange={(e) =>
                    setRadarSensitivity(parseFloat(e.target.value))
                  }
                  className="w-full"
                />
                <span className="text-xs text-gray-500">
                  {Math.round(radarSensitivity * 100)}%
                </span>
              </div>
              <div className="text-center">
                <div className="text-gray-400 mb-1">Defense Mode</div>
                <div
                  className={`font-bold ${
                    defenseMode === "AGGRESSIVE"
                      ? "text-red-400"
                      : defenseMode === "DEFENSIVE"
                      ? "text-blue-400"
                      : defenseMode === "AI_AUTONOMOUS"
                      ? "text-green-400"
                      : "text-green-400"
                  }`}
                >
                  {defenseMode}
                </div>
              </div>
              <div className="text-right">
                <div className="text-gray-400 mb-1">System Uptime</div>
                <div className="font-mono text-green-400">
                  {formatUptime(systemMetrics.systemUptime)}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Control Panels */}
        <div className="lg:col-span-2 space-y-6">
          {/* Threat Status Panel */}
          <div className="bg-gray-800 border border-green-500 rounded-lg p-4 min-h-[140px]">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <Target className="w-5 h-5" />
              Threat Analysis
            </h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>Total Threats:</span>
                  <span className="text-white font-mono">
                    {threatStats.total}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Detected:</span>
                  <span className="text-yellow-400 font-mono">
                    {threatStats.detected}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Interceptors:</span>
                  <span className="text-blue-400 font-mono">
                    {threatStats.interceptors}
                  </span>
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-red-400">Critical:</span>
                  <span className="text-red-400 font-mono">
                    {threatStats.critical}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-orange-400">High:</span>
                  <span className="text-orange-400 font-mono">
                    {threatStats.high}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-green-400">Destroyed:</span>
                  <span className="text-green-400 font-mono">
                    {threatStats.destroyed}
                  </span>
                </div>
              </div>
            </div>
            <div className="mt-2 pt-2 border-t border-gray-600">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between">
                  <span>Misses:</span>
                  <span className="text-red-400 font-mono">
                    {threatStats.escaped}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Accuracy:</span>
                  <span className="text-green-400 font-mono">
                    {Math.round((threatStats.avgConfidence || 0) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* AI Analysis Panel */}
          <div className="bg-gray-800 border border-green-500 rounded-lg p-4 min-h-[240px]">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Neural Analysis
            </h3>
            <div className="space-y-2 min-h-[180px] max-h-[180px] overflow-y-auto">
              {prioritizedThreats.map((threat) => (
                <div
                  key={threat.id}
                  className={`text-xs border-l-4 pl-2 p-2 rounded-r ${
                    threat.threatLevel === "CRITICAL"
                      ? "border-red-500 bg-red-900/20"
                      : threat.threatLevel === "HIGH"
                      ? "border-orange-500 bg-orange-900/20"
                      : threat.threatLevel === "MEDIUM"
                      ? "border-yellow-500 bg-yellow-900/20"
                      : "border-green-500 bg-green-900/20"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-semibold">{threat.type}</span>
                    <div className="flex gap-2">
                      <span className="text-red-400">P{threat.priority}</span>
                      <span className="text-blue-400">
                        {Math.round((threat.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between text-gray-400 mt-1">
                    <span>
                      Range: {Math.round(threat.distanceToCenter || 0)}m
                    </span>
                    <span>
                      ETA: {Math.round(threat.estimatedImpactTime || 0)}s
                    </span>
                  </div>
                  <div className="text-gray-500 text-xs mt-1">
                    Pattern: {threat.behaviorPattern} | Level:{" "}
                    {threat.threatLevel}
                  </div>
                </div>
              ))}
              {prioritizedThreats.length === 0 && (
                <div className="text-gray-500 text-sm italic text-center py-8">
                  No threats detected
                </div>
              )}
            </div>
          </div>

          {/* System Metrics */}
          <div className="bg-gray-800 border border-green-500 rounded-lg p-4 min-h-[160px]">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Performance Metrics
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>System Status:</span>
                <span
                  className={`font-mono ${
                    isRunning ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {isRunning ? "ACTIVE" : "STANDBY"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Threats Detected:</span>
                <span className="font-mono text-yellow-400">
                  {systemMetrics.threatsDetected}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Successful Intercepts:</span>
                <span className="font-mono text-green-400">
                  {systemMetrics.successfulInterceptions}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Success Rate:</span>
                <span className="font-mono text-blue-400">
                  {systemMetrics.threatsDetected > 0
                    ? Math.round(
                        (systemMetrics.successfulInterceptions /
                          systemMetrics.threatsDetected) *
                          100
                      )
                    : 0}
                  %
                </span>
              </div>
              <div className="flex justify-between">
                <span>Radar Angle:</span>
                <span className="font-mono">
                  {Math.round(((radarSweep * 180) / Math.PI) % 360)}°
                </span>
              </div>
              <div className="flex justify-between">
                <span>AI Status:</span>
                <span
                  className={`font-mono ${
                    aiEnabled ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {aiEnabled ? "ACTIVE" : "OFFLINE"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>AI Accuracy:</span>
                <span className="font-mono text-blue-400">
                  {Math.round(systemMetrics.aiAccuracy * 100)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Training Cycles:</span>
                <span className="font-mono text-purple-400">
                  {systemMetrics.modelTrainingCount}
                </span>
              </div>
            </div>
          </div>

          {/* Enhanced Legend */}
          <div className="bg-gray-800 border border-green-500 rounded-lg p-4 min-h-[180px]">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Threat Classification
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full flex-shrink-0"></div>
                <span>Missile (Critical Priority)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-gray-600 rounded-full flex-shrink-0"></div>
                <span>Stealth (High Priority)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full flex-shrink-0"></div>
                <span>Aircraft (Medium)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-300 rounded-full flex-shrink-0"></div>
                <span>Drone/Swarm (Low-Med)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-400 rounded-full flex-shrink-0"></div>
                <span>Interceptor</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-400 rounded-full flex-shrink-0"></div>
                <span>Destroyed Target</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-400 rounded-full flex-shrink-0"></div>
                <span>Miss/Explosion</span>
              </div>
              <div className="mt-3 pt-2 border-t border-gray-600 text-xs text-gray-400">
                <div>Defense Modes:</div>
                <div className="mt-1 space-y-1">
                  <div>
                    <span className="text-blue-400">Defensive:</span>{" "}
                    Conservative targeting
                  </div>
                  <div>
                    <span className="text-green-400">Adaptive:</span> Balanced
                    response
                  </div>
                  <div>
                    <span className="text-red-400">Aggressive:</span> Maximum
                    interception
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AirDefenseSimulation;
