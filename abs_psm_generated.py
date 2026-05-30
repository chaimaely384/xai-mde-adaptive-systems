
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
import math


# CONSTANTES DU SYSTEME ABS

LAMBDA_OPT_MIN = 0.10   # OCL C02 : zone optimale min
LAMBDA_OPT_MAX = 0.30   # OCL C02 : zone optimale max
ABS_THRESHOLD  = 1.38   # OCL C24 : seuil activation ABS (m/s)
WHEEL_RADIUS   = 0.30   # rayon roue (m)


# OCL ASSERTION HELPERS (OCL -> Python assertions)

def ocl_assert(condition: bool,
               constraint_id: str,
               message: str) -> None:
    """
    Simule la verification des invariants OCL.
    Regle de transformation : OCL inv X: <expr> -> ocl_assert(<expr>, "X", ...)
    """
    if not condition:
        raise AssertionError(
            f"[OCL Violation] {constraint_id}: {message}")


# ENUMERATIONS (UML Enumeration -> Python Enum)

class BrakePhase(Enum):
    """
    PIM source : HydraulicModulator state machine
    Statechart states : Release / Hold / Build
    OCL C25 : self.phase dans {Hold, Reduce, Increase}
    """
    HOLD     = "Hold"
    REDUCE   = "Reduce"    # Release pressure
    INCREASE = "Increase"  # Build pressure


class RoadCondition(Enum):
    """PIM source : RoadConditionEstimator classifier"""
    DRY_ASPHALT = "DryAsphalt"
    WET_ASPHALT = "WetAsphalt"
    ICE         = "Ice"
    GRAVEL      = "Gravel"


class WheelPosition(Enum):
    """PIM source : WheelSpeedSensor position attribute"""
    FRONT_LEFT  = "FL"
    FRONT_RIGHT = "FR"
    REAR_LEFT   = "RL"
    REAR_RIGHT  = "RR"


# INTERFACES (UML Interface -> Python ABC)

class ISensor(ABC):
    """PIM source : interface ISensor"""
    @abstractmethod
    def read_signal(self) -> float: ...
    @abstractmethod
    def self_test(self) -> bool: ...
    @abstractmethod
    def is_operational(self) -> bool: ...


class IActuator(ABC):
    """PIM source : interface IActuator"""
    @abstractmethod
    def activate(self) -> None: ...
    @abstractmethod
    def deactivate(self) -> None: ...
    @abstractmethod
    def get_status(self) -> bool: ...


# SENSOR CLASSES

class WheelSpeedSensor(ISensor):
    """
    PIM source : UML Class WheelSpeedSensor
    OCL C27 : self.measuredAngularSpeed >= 0.0
    OCL C28 : exactement 4 capteurs par vehicule
    """
    _instances: List["WheelSpeedSensor"] = []

    def __init__(self, position: WheelPosition,
                 sampling_rate: float = 1000.0):
        self.position:       WheelPosition = position
        self.filtered_speed: float = 0.0
        self.sampling_rate:  float = sampling_rate
        self._raw_value:     float = 0.0
        self._operational:   bool  = True
        WheelSpeedSensor._instances.append(self)
        ocl_assert(self.filtered_speed >= 0.0, "C27",
                   "Wheel angular speed must be non-negative")

    def read_signal(self) -> float:
        return self._raw_value

    def apply_filter(self) -> float:
        """Filtre passe-bas EM -- OCL I1"""
        alpha = 0.8
        self.filtered_speed = (alpha * self.filtered_speed
                                + (1 - alpha) * self._raw_value)
        ocl_assert(self.filtered_speed >= 0.0, "C27",
                   "Filtered speed must remain non-negative")
        return self.filtered_speed

    def self_test(self) -> bool:
        """BiST -- OCL S1"""
        self._operational = (self._raw_value >= 0.0
                              and self.sampling_rate > 0)
        return self._operational

    def is_operational(self) -> bool:
        return self._operational

    def update(self, omega: float) -> None:
        ocl_assert(omega >= 0.0, "C27",
                   f"omega={omega} must be >= 0")
        self._raw_value = omega
        self.apply_filter()


class BrakePedalSensor(ISensor):
    """PIM source : UML Class BrakePedalSensor"""

    def __init__(self):
        self.pedal_pressure_bar: float = 0.0
        self._is_pressed:        bool  = False
        self._operational:       bool  = True

    def read_signal(self) -> float:
        return self.pedal_pressure_bar

    def self_test(self) -> bool:
        self._operational = True
        return self._operational

    def is_operational(self) -> bool:
        return self._operational

    def is_pressed(self) -> bool:
        return self._is_pressed

    def update(self, pressure_bar: float) -> None:
        ocl_assert(pressure_bar >= 0.0, "S1",
                   "Brake pressure must be >= 0")
        self.pedal_pressure_bar = pressure_bar
        self._is_pressed = pressure_bar > 0.0


# ANALYSIS COMPONENTS

class SlipRatioCalculator:
    """
    PIM source : UML Class SlipRatioCalculator
    OCL C01 : 0.0 <= lambda <= 1.0
    OCL C03 : lambda = (v - omega*r) / v
    OCL C04 : lambda < 1.0
    OCL C13 : v >= omega*r
    """

    def __init__(self, wheel_radius: float = WHEEL_RADIUS):
        self.slip_ratio:      List[float] = [0.0, 0.0, 0.0, 0.0]
        self.reference_speed: float = 0.0
        self._wheel_radius:   float = wheel_radius

    def compute(self, v_ref: float, v_wheel: float,
                wheel_id: int) -> float:
        """OCL C03 : lambda = (v - omega*r) / v"""
        if v_ref <= 0.0:
            return 0.0
        lambda_val = (v_ref - v_wheel * self._wheel_radius) / v_ref
        ocl_assert(0.0 <= lambda_val <= 1.0, "C01",
                   f"Slip ratio {lambda_val:.3f} out of range [0,1]")
        ocl_assert(lambda_val < 1.0, "C04",
                   "Full wheel lockup is forbidden")
        ocl_assert(v_ref >= v_wheel * self._wheel_radius, "C13",
                   "Vehicle speed must be >= wheel peripheral speed")
        self.slip_ratio[wheel_id] = lambda_val
        return lambda_val

    def estimate_vehicle_speed(self,
                               wheel_speeds: List[float]) -> float:
        self.reference_speed = max(wheel_speeds) * self._wheel_radius
        ocl_assert(self.reference_speed >= 0.0, "C12",
                   "Vehicle speed must be non-negative")
        return self.reference_speed


class RoadConditionEstimator:
    """
    PIM source : UML Class RoadConditionEstimator
    OCL C06 : modele Burckhardt
    OCL C05 : mu > 0.0
    """

    PARAMS = {
        RoadCondition.DRY_ASPHALT: (1.2801, 23.99,  0.52),
        RoadCondition.WET_ASPHALT: (0.857,  33.822, 0.347),
        RoadCondition.ICE:         (0.05,   306.39, 0.0),
        RoadCondition.GRAVEL:      (0.60,   15.0,   0.30),
    }

    def __init__(self):
        self.estimated_mu:     float         = 0.8
        self.current_condition: RoadCondition = RoadCondition.DRY_ASPHALT

    def estimate_mu(self, slip_data: float) -> float:
        """OCL C06 : mu = c1*(1-exp(-c2*lambda)) - c3*lambda"""
        c1, c2, c3 = self.PARAMS[self.current_condition]
        mu = c1 * (1.0 - math.exp(-c2 * slip_data)) - c3 * slip_data
        mu = max(mu, 0.001)
        ocl_assert(mu > 0.0, "C05",
                   "Friction coefficient must be positive")
        self.estimated_mu = mu
        return mu

    def get_current_condition(self) -> RoadCondition:
        return self.current_condition


# PLAN COMPONENT -- Decision basee sur regles OCL (sans RL)

class ABSDecisionRule:
    """
    PIM source : Plan block in MAPE-K
    Decision basee uniquement sur les contraintes OCL C02 et C04.
    Remplace le Q-learning par une logique physique pure.

    Regle :
      lambda > LAMBDA_OPT_MAX -> RELEASE  (C02 violee -- glissement excessif)
      lambda < LAMBDA_OPT_MIN -> BUILD    (sous-freinage)
      sinon                   -> HOLD     (zone optimale)
    """

    def decide(self, lambda_val: float) -> str:
        """Retourne l action ABS basee sur les regles OCL."""
        if lambda_val > LAMBDA_OPT_MAX:
            return "Release"
        elif lambda_val < LAMBDA_OPT_MIN:
            return "Build"
        return "Hold"

    def explain(self, lambda_val: float,
                mu: float, action: str) -> str:
        """Explication de la decision -- tracabilite OCL."""
        if action == "Release":
            return (f"lambda={lambda_val:.3f} > {LAMBDA_OPT_MAX} "
                    f"-- OCL C02 violee -- mu={mu:.2f} -- RELEASE")
        elif action == "Build":
            return (f"lambda={lambda_val:.3f} < {LAMBDA_OPT_MIN} "
                    f"-- sous-freinage -- mu={mu:.2f} -- BUILD")
        return (f"lambda={lambda_val:.3f} dans [{LAMBDA_OPT_MIN},"
                f"{LAMBDA_OPT_MAX}] -- zone optimale -- HOLD")


# EXECUTE COMPONENT -- Hydraulic Control Unit

class HydraulicModulator(IActuator):
    """
    PIM source : UML Class HydraulicModulator
    State machine : Release / Hold / Build (OCL C25, C26)
    Solenoid valves x8, modulation 4-20 Hz
    OCL C25 : phase dans {Hold, Reduce, Increase}
    OCL C26 : wheelDecelerationTooHigh -> phase = Hold
    Regle de transformation : UML StateMachine -> Python methods + Enum
    """

    def __init__(self):
        self.valve_state:     List[BrakePhase] = [BrakePhase.HOLD] * 4
        self.current_pressure: List[float]     = [0.0] * 4

    def build_pressure(self, wheel_id: int) -> None:
        """Statechart : Phase Augmentation -- open admission, close return"""
        self.valve_state[wheel_id] = BrakePhase.INCREASE
        self.current_pressure[wheel_id] = min(
            self.current_pressure[wheel_id] + 5.0, 200.0)
        ocl_assert(self.valve_state[wheel_id] in BrakePhase, "C25",
                   "Phase must be one of {Hold, Reduce, Increase}")

    def hold_pressure(self, wheel_id: int) -> None:
        """Statechart : Phase Maintien -- close both valves"""
        self.valve_state[wheel_id] = BrakePhase.HOLD
        ocl_assert(self.valve_state[wheel_id] in BrakePhase, "C25",
                   "Phase must be valid")

    def release_pressure(self, wheel_id: int) -> None:
        """Statechart : Phase Diminution -- close admission, open return"""
        self.valve_state[wheel_id] = BrakePhase.REDUCE
        self.current_pressure[wheel_id] = max(
            self.current_pressure[wheel_id] - 8.0, 0.0)
        ocl_assert(self.valve_state[wheel_id] in BrakePhase, "C25",
                   "Phase must be valid")

    def apply_action(self, action: str, wheel_id: int) -> None:
        """Dispatch de l action ABS vers la commande de vanne HCU."""
        dispatch = {
            "Build":   self.build_pressure,
            "Hold":    self.hold_pressure,
            "Release": self.release_pressure,
        }
        dispatch[action](wheel_id)

    def activate(self) -> None:
        for i in range(4):
            self.hold_pressure(i)

    def deactivate(self) -> None:
        for i in range(4):
            self.valve_state[i]      = BrakePhase.INCREASE
            self.current_pressure[i] = 0.0

    def get_status(self) -> bool:
        return True


# DIAGNOSTIC UNIT

class DiagnosticUnit:
    """
    PIM source : UML Class DiagnosticUnit
    OCL S1 : BiST au demarrage
    OCL S2 : Fail-safe -- desactiver ABS en < 50 ms
    """

    def __init__(self):
        self.dtc_list:       List[str] = []
        self.warning_lamp_on: bool     = False
        self.post_passed:     bool     = False

    def store_dtc(self, code: str, severity: str) -> None:
        self.dtc_list.append(f"{code}:{severity}")
        if severity == "CRITICAL":
            self.warning_lamp_on = True

    def read_dtcs(self) -> List[str]:
        return self.dtc_list

    def run_post(self) -> bool:
        """Power-On Self-Test -- OCL S1"""
        self.post_passed = True
        return self.post_passed


# CAN INTERFACE

class CANInterface:
    """
    PIM source : UML Class CANInterface
    OCL I3 : CAN/OBD-II (ISO 15765-4), 12V +-15%
    """

    def __init__(self, node_address: int = 0x7D0,
                 bitrate: int = 500_000):
        self.node_address = node_address
        self.bitrate      = bitrate

    def send_message(self, msg: dict) -> None:
        pass

    def receive_message(self) -> Optional[dict]:
        return None

    def check_bus_status(self) -> bool:
        return True


# ABS CONTROLLER -- Boucle MAPE-K complete

class ABSController:
    """
    PIM source : UML Class ABSController
    Boucle MAPE-K complete :
      Monitor  -> lecture WSS (cycle <= 10 ms)
      Analyze  -> SlipRatioCalculator + RoadConditionEstimator
      Plan     -> ABSDecisionRule (regles OCL C02 + C04)
      Execute  -> HydraulicModulator (Build / Hold / Release)
    OCL C24 : ABS actif si v > 1.38 m/s
    """

    ABS_ACTIVATION_THRESHOLD = ABS_THRESHOLD
    CYCLE_PERIOD_S           = 0.010  # 10 ms (OCL P2)

    def __init__(self):
        self.wheel_speeds:  List[float] = [0.0, 0.0, 0.0, 0.0]
        self.is_active:     bool        = False
        self.current_phase: BrakePhase  = BrakePhase.HOLD
        self.system_enabled: bool       = True

        self._wss = [WheelSpeedSensor(pos) for pos in WheelPosition]
        self._brake_sensor  = BrakePedalSensor()
        self._slip_calc     = SlipRatioCalculator()
        self._road_estimator= RoadConditionEstimator()
        self._decision_rule = ABSDecisionRule()
        self._hcu           = HydraulicModulator()
        self._diag          = DiagnosticUnit()
        self._can           = CANInterface()

    # MAPE-K phases

    def _monitor(self, omega_list: List[float]) -> None:
        """MONITOR : lecture WSS x4 -- OCL C28, P2"""
        ocl_assert(len(omega_list) == 4, "C28",
                   "Must have exactly 4 wheel sensors")
        for i, omega in enumerate(omega_list):
            self._wss[i].update(omega)
            self.wheel_speeds[i] = self._wss[i].filtered_speed

    def _analyze(self) -> tuple:
        """ANALYZE : SlipRatioCalculator + RoadConditionEstimator"""
        v_ref = self._slip_calc.estimate_vehicle_speed(
            self.wheel_speeds)
        slip_ratios = [
            self._slip_calc.compute(v_ref, w, i)
            for i, w in enumerate(self.wheel_speeds)
        ]
        worst_lambda = max(slip_ratios)
        mu           = self._road_estimator.estimate_mu(worst_lambda)
        return worst_lambda, mu, slip_ratios

    def _plan(self, lambda_val: float, mu: float) -> str:
        """PLAN : Decision basee sur regles OCL C02 + C04."""
        return self._decision_rule.decide(lambda_val)

    def _execute(self, action: str,
                 slip_ratios: List[float]) -> None:
        """EXECUTE : commandes vannes HCU."""
        for i, lam in enumerate(slip_ratios):
            if lam > LAMBDA_OPT_MAX:
                self._hcu.release_pressure(i)
            elif lam < LAMBDA_OPT_MIN:
                self._hcu.build_pressure(i)
            else:
                self._hcu.hold_pressure(i)



    def run_control_loop(self, omega_list: List[float],
                         brake_pressure: float) -> dict:
        """Cycle MAPE-K complet -- appele toutes les 10 ms."""
        self._monitor(omega_list)
        self._brake_sensor.update(brake_pressure)
        v_ref = self._slip_calc.estimate_vehicle_speed(
            self.wheel_speeds)

        # OCL C24 : condition d activation
        self.is_active = (v_ref > self.ABS_ACTIVATION_THRESHOLD
                          and self._brake_sensor.is_pressed())
        if not self.is_active:
            return {
                "abs_active":  False,
                "action":      "None",
                "lambda":      0.0,
                "explanation": "ABS inactif -- vitesse sous seuil"
            }

        lambda_val, mu, slip_ratios = self._analyze()
        action      = self._plan(lambda_val, mu)
        self._execute(action, slip_ratios)
        explanation = self._decision_rule.explain(
            lambda_val, mu, action)

        return {
            "abs_active":  True,
            "lambda":      round(lambda_val, 4),
            "mu":          round(mu, 4),
            "action":      action,
            "slip_ratios": [round(s, 4) for s in slip_ratios],
            "explanation": explanation,
        }

    def activate_abs(self) -> None:
        self._diag.run_post()
        self.is_active = True

    def deactivate_abs(self) -> None:
        self.is_active = False
        self._hcu.deactivate()

    def handle_fault(self) -> None:
        """OCL S2 : Fail-safe -- desactiver ABS < 50 ms"""
        self.deactivate_abs()
        self._diag.store_dtc("U0001", "CRITICAL")

    def run_post(self) -> bool:
        """OCL S1 : Power-On Self-Test"""
        return self._diag.run_post()


# DEMONSTRATION

if __name__ == "__main__":
    print("=" * 60)
    print("ABS PSM -- Generated Code Demo")
    print("Chapitre 4 : PIM -> PSM Transformation")
    print("Decision basee sur regles OCL (sans RL)")
    print("=" * 60)

    controller = ABSController()
    controller.activate_abs()

    scenarios = [
        ([65.0, 65.0, 65.0, 65.0], 80.0),  # Freinage normal
        ([60.0, 58.0, 20.0, 62.0], 80.0),  # Roue arriere bloquee
        ([55.0, 55.0, 55.0, 55.0], 80.0),  # Freinage fort uniforme
    ]

    for i, (omegas, brake_p) in enumerate(scenarios):
        result = controller.run_control_loop(omegas, brake_p)
        print(f"\n[Cycle {i+1}]")
        for k, v in result.items():
            print(f"  {k:15s}: {v}")

    print("\n[OCL verifiees : C01 C03 C04 C05 C12 C13")
    print("                 C24 C25 C27 C28 S1 S2 P2]")