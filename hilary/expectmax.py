from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import factorial


class EM:
    """Finds the mixture distribution
    that explains the statistics of distances"""

    def __init__(
        self, l, h, model=326713, howmany=10, positives="geometric", cdfs=None
    ):
        self.l = int(l)
        self.h = h.astype(int)[: self.l + 1]
        self.b = np.arange(self.l + 1, dtype=int)
        self.model = model
        self.cdfs = cdfs
        self.const_p0 = self.readNull()
        self.howmany = howmany
        self.positives = positives

    def readNull(self):
        """Read estimated null distribution"""
        cdf = self.cdfs.loc[self.cdfs["l"] == self.l].values[0, 1 : self.l + 1]
        return np.diff(cdf, prepend=[0], append=[1])

    def discreteExpectation(self, theta):
        """Calculate membership probabilities"""
        rho, mu = theta
        if self.positives == "geometric":
            P1 = rho / (mu + 1) * (mu / (mu + 1)) ** self.b
        elif self.positives == "poisson":
            P1 = rho * mu**self.b * np.exp(-mu) / factorial(self.b)
        P0 = (1 - rho) * self.const_p0[self.b]
        return np.array([P1, P0]) / (P1 + P0 + 1e-5)

    def dicreteMaximization(self, theta):
        """Maximize current likelihood"""
        P1, P0 = self.discreteExpectation(theta)
        P1Sum, P0Sum = (self.h * P1).sum(), (self.h * P0).sum()
        rho = min(P1Sum / (P1Sum + P0Sum), 1.0)
        mu = np.dot(self.h * P1, self.b) / (P1Sum + 1e-5)
        return rho, mu

    def discreteEM(self):
        """EM with discrete model:
        P1 geometric or Poisson with mean mu
        P0 estimated with Ppost"""
        mu = 0.02 * self.l
        rho = self.h[0] / sum(self.h) * (1 + mu)
        theta = (max(min(1.0, rho), 0.1), mu)
        for _ in range(self.howmany):
            # if np.isnan(theta[0]) or np.isnan(theta[1]):
            #    print("nan appears at ", i - 1)
            #    return theta
            theta = self.dicreteMaximization(theta)
        return theta

    def discreteMix(self, theta):
        """Evaluate mixture distribution"""
        rho, mu = theta
        if self.positives == "geometric":
            return (
                rho / (mu + 1) * (mu / (mu + 1)) ** self.b
                + (1 - rho) * self.const_p0[self.b]
            )
        if self.positives == "poisson":
            return (
                rho * mu**self.b * np.exp(-mu) / factorial(self.b)
                + (1 - rho) * self.const_p0[self.b]
            )

    def error(self, theta, threshold=0.15, error="rrmse"):
        """Estimate goodness of fit
        By default returns rescaled root MSE"""
        y1 = self.h / sum(self.h)
        y2 = self.discreteMix(theta)
        mask = (y1 > 0) * (self.b < threshold * self.l)
        y1m, y2m = y1[mask], y2[mask]
        logy1, logy2 = np.log(y1m + 1e-5), np.log(y2m + 1e-5)

        mse = ((y1m - y2m) ** 2).sum() / (mask.sum() + 1e-5)
        rmse = np.sqrt(mse)
        msle = ((logy1 - logy2) ** 2).sum() / (mask.sum() + 1e-5)
        rmsle = np.sqrt(msle)
        mae = np.abs(y1m - y2m).sum() / (mask.sum() + 1e-5)
        dkl = (y2m * (logy2 - logy1)).sum() / np.log(2)
        if error == "rmse":
            return rmse
        elif error == "rmsle":
            return rmsle
        elif error == "mae":
            return mae
        elif error == "dkl":
            return dkl
        elif error == "rrmse" and theta[0] > 0:
            return rmse / (theta[0] + 1e-5)
        else:
            return False
