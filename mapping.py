from architectures.encoders.cnn import CNNEncoder
from architectures.encoders.cnn_grouped import CNNGroupedEncoder
from architectures.encoders.special.phaser import PhaserEncoder
from architectures.encoders.special.diversify import DiversifyEncoder
from architectures.encoders.cnn_lstm import CNNLSTMEncoder
from architectures.encoders.transformer import TransformerEncoder

from architectures.models.early import EarlyFusionModel
from architectures.models.grouped.middle import MiddleFusionModel, MiddleEqualFusionModel, MiddleNormFusionModel, MiddleEqualNormFusionModel
from architectures.models.grouped.late import LateFusionModel, LateFusionWeightModel
from architectures.models.looped.middle import MiddleFusionModel as MiddleFusionModelL
from architectures.models.looped.late import LateFusionModel as LateFusionModelL
from architectures.models.looped.ensemble import EarlyFusionEnsemble
from architectures.models.special.diversify import DiversifyModel

from trainers.erm.early import EarlyERMTrainer
from trainers.erm.late import LateERMTrainer, LateERMTrainerL
from trainers.erm.ensemble import EnsembleERMTrainer
from trainers.dg.mmd import MMDTrainer
from trainers.dg.vrex import VREXTrainer
from trainers.dg.nnr import NNRTrainer
from trainers.dg.irm import IRMTrainer
from trainers.dg.groupdro import GroupDROTrainer
from trainers.dg.diversify import DiversifyTrainer


def get_classes(encoder, fusion, algorithm):

    if encoder == "cnn":
        if fusion == "early":
            if algorithm == "erm":
                return CNNEncoder, EarlyFusionModel, EarlyERMTrainer
            elif algorithm == "mmd":
                return CNNEncoder, EarlyFusionModel, MMDTrainer
            elif algorithm == "vrex":
                return CNNEncoder, EarlyFusionModel, VREXTrainer
            elif algorithm == "nnr":
                return CNNEncoder, EarlyFusionModel, NNRTrainer
            elif algorithm == "irm":
                return CNNEncoder, EarlyFusionModel, IRMTrainer
            elif algorithm == "groupdro":
                return CNNEncoder, EarlyFusionModel, GroupDROTrainer
            elif algorithm == "diversify":
                return DiversifyEncoder, DiversifyModel, DiversifyTrainer
            elif algorithm == "phaser":
                return PhaserEncoder, EarlyFusionModel, EarlyERMTrainer
            elif algorithm == "erm_ens":
                return CNNEncoder, EarlyFusionEnsemble, EnsembleERMTrainer
        elif fusion == "middle":
            if algorithm == "erm":
                return CNNGroupedEncoder, MiddleFusionModel, EarlyERMTrainer
        elif fusion == "middle_e":
            if algorithm == "erm":
                return CNNGroupedEncoder, MiddleEqualFusionModel, EarlyERMTrainer
        elif fusion == "middle_n":
            if algorithm == "erm":
                return CNNGroupedEncoder, MiddleNormFusionModel, EarlyERMTrainer
        elif fusion == "middle_en":
            if algorithm == "erm":
                return CNNGroupedEncoder, MiddleEqualNormFusionModel, EarlyERMTrainer
        elif fusion == "late":
            if algorithm == "erm":
                return CNNGroupedEncoder, LateFusionModel, LateERMTrainer
        elif fusion == "late_w":
            if algorithm == "erm":
                return CNNGroupedEncoder, LateFusionWeightModel, LateERMTrainer

    elif encoder == "cnn_lstm":
        if fusion == "early":
            if algorithm == "erm":
                return CNNLSTMEncoder, EarlyFusionModel, EarlyERMTrainer
        elif fusion == "middle":
            if algorithm == "erm":
                return CNNLSTMEncoder, MiddleFusionModelL, EarlyERMTrainer
        elif fusion == "late":
            if algorithm == "erm":
                return CNNLSTMEncoder, LateFusionModelL, LateERMTrainerL

    elif encoder == "transformer":
        if fusion == "early":
            if algorithm == "erm":
                return TransformerEncoder, EarlyFusionModel, EarlyERMTrainer
        elif fusion == "middle":
            if algorithm == "erm":
                return TransformerEncoder, MiddleFusionModelL, EarlyERMTrainer
        elif fusion == "late":
            if algorithm == "erm":
                return TransformerEncoder, LateFusionModelL, LateERMTrainerL

    raise ValueError(f"invalid combo: {encoder} - {fusion} - {algorithm}")
