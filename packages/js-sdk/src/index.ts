export type FrameFormat = "rgb" | "bgr" | "grayscale" | "jpeg" | "h264";

export type OverlayType =
  | "text"
  | "icon"
  | "bounding_box"
  | "reticle"
  | "notification"
  | "image"
  | "region";

export type FrameMessage = {
  frameId: string;
  timestampMs: number;
  width: number;
  height: number;
  format: FrameFormat | string;
  data?: Uint8Array;
  metadata?: Record<string, unknown>;
};

export type PoseMessage = {
  timestampMs: number;
  orientation: [number, number, number, number];
  position?: [number, number, number];
  coordinateSystem?: string;
  angularVelocity?: [number, number, number];
  linearAcceleration?: [number, number, number];
  metadata?: Record<string, unknown>;
};

export type OverlayMessage = {
  overlayId: string;
  type: OverlayType | string;
  timestampMs: number;
  frameId?: string;
  expiresAtMs?: number;
  geometry?: Record<string, unknown>;
  style?: Record<string, unknown>;
  payload: Record<string, unknown>;
};

export type DeviceEvent = {
  type: "input" | "app" | "status" | "control" | string;
  timestampMs: number;
  payload?: Record<string, unknown>;
};

export type DeviceStatus = {
  connected: boolean;
  streaming: boolean;
  timestampMs: number;
  batteryPercent?: number;
  transport?: string;
  metadata?: Record<string, unknown>;
};

export type Device = {
  connect(): Promise<Device> | Device;
  getFrame(): Promise<FrameMessage> | FrameMessage;
  getPose(): Promise<PoseMessage> | PoseMessage;
  renderOverlay(overlay: OverlayMessage): Promise<void> | void;
  sendEvent(event: DeviceEvent): Promise<void> | void;
  startStream(): Promise<void> | void;
  stopStream(): Promise<void> | void;
  status(): Promise<DeviceStatus> | DeviceStatus;
};

const nowMs = () => Date.now();

export class SimulatorDevice implements Device {
  connected = false;
  streaming = false;
  overlays: OverlayMessage[] = [];
  events: DeviceEvent[] = [];
  private frameCounter = 0;

  constructor(
    public width = 1280,
    public height = 720,
    public format: FrameFormat | string = "rgb",
  ) {}

  connect(): this {
    this.connected = true;
    return this;
  }

  getFrame(): FrameMessage {
    this.requireConnected();
    this.frameCounter += 1;
    return {
      frameId: `sim-${this.frameCounter}`,
      timestampMs: nowMs(),
      width: this.width,
      height: this.height,
      format: this.format,
      metadata: { source: "simulator" },
    };
  }

  getPose(): PoseMessage {
    this.requireConnected();
    return {
      timestampMs: nowMs(),
      orientation: [0, 0, 0, 1],
      position: [0, 0, 0],
      coordinateSystem: "right-handed-y-up",
      metadata: { source: "simulator" },
    };
  }

  renderOverlay(overlay: OverlayMessage): void {
    this.requireConnected();
    this.overlays.push(overlay);
  }

  sendEvent(event: DeviceEvent): void {
    this.requireConnected();
    this.events.push(event);
  }

  startStream(): void {
    this.requireConnected();
    this.streaming = true;
  }

  stopStream(): void {
    this.streaming = false;
  }

  status(): DeviceStatus {
    return {
      connected: this.connected,
      streaming: this.streaming,
      timestampMs: nowMs(),
      transport: "simulator",
    };
  }

  private requireConnected(): void {
    if (!this.connected) {
      throw new Error("device is not connected");
    }
  }
}

export type PluginContext = {
  device: Device;
  state: Record<string, unknown>;
  emit(overlay: OverlayMessage): Promise<void> | void;
};

export type Plugin = {
  setup?(context: PluginContext): Promise<void> | void;
  onFrame?(frame: FrameMessage, context: PluginContext): Promise<void> | void;
  onPose?(pose: PoseMessage, context: PluginContext): Promise<void> | void;
  onEvent?(event: DeviceEvent, context: PluginContext): Promise<void> | void;
  teardown?(context: PluginContext): Promise<void> | void;
};

export class PlaceholderTransport {
  constructor(public name: "usb" | "ble" | "wifi" | string) {}

  connect(): never {
    throw new Error(`${this.name} transport is a skeleton placeholder until hardware details are known`);
  }
}
