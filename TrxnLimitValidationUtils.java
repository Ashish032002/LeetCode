package com.cams.core.rules.utils;

import com.cams.core.api.requests.RuleEngineRequest;
import com.cams.core.api.requests.RuleEngineRequestContext;
import com.cams.core.domain.*;
import com.cams.core.model.TransactionType;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.ZonedDateTime;
import java.util.List;

public class TrxnLimitValidationUtils {

    private TrxnLimitValidationUtils() {}

    private static final BigDecimal ZERO = BigDecimal.ZERO;
    private static final BigDecimal HUNDRED = BigDecimal.valueOf(100);

    // ---- COMPUTE METHODS + PUBLIC WRAPPERS (called from DRL rules) ----

    private static Object[] computePurchaseAmountInternal(RuleEngineRequest req) {
        // [0] baseGuardPassed, [1] signGuardPassed, [2] effectiveMin, [3] effectiveMax,
        // [4] totalPurchaseAmount, [5] minAmountTotal, [6] multiples, [7] currentPurchaseAmountRounded,
        // [8] skipMinCheck, [9] skipMultiplesCheck, [10] tooManySchemeRows, [11] tooManySubFundRows,
        // [12] tooManyFundRows, [13] schemeCode, [14] parentSubFundCode, [15] parentFundCode, [16] txnTypeName
        Object[] result = new Object[] {
                Boolean.FALSE, Boolean.FALSE, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO,
                Boolean.FALSE, Boolean.FALSE, Boolean.FALSE, Boolean.FALSE,
                Boolean.FALSE, "", "", "", ""
        };

        if (req == null || req.getContext() == null) return result;

        BigDecimal nav = getNav(req);
        Long autoTrxnNo = computeAutoTrxnNo(req);
        String unitsAmountFlag = computeUnitsAmountFlag(req, nav);
        String subTrxnType = computeSubTrxnType(req);
        String newPurchaseTypeFlag = computeNewPurchaseTypeFlag(req);

        Decuple<String, String, String, Void, Void, Void, Void, Void, Void, Void> aggreg = computeAggregFlags(req);
        String aggregPlusTxForLimitCheck = aggreg.getSecond();

        Decuple<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void> swapFreq = computeSwapAndFrequency(req, autoTrxnNo);
        String swapOption = swapFreq.getFirst();

        ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
        String sourceProgram = cdtp != null ? cdtp.getSourceProgram() : null;
        String motherName = cdtp != null ? cdtp.getMotherName() : null;
        String spouseFirstName = cdtp != null ? cdtp.getSpouseFirstName() : null;
        String onlineUserFlag = req.getUserMaster() != null ? req.getUserMaster().getOnlineUser() : null;

        String userType = getUserType(req);
        String dpIdRequired = getDpIdRequired(req);

        String applyLimits = computeApplyLimits(req, autoTrxnNo, swapOption, sourceProgram,
                userType, dpIdRequired, motherName);
        if ("X".equals(applyLimits)) return result;

        Decuple<String, BigDecimal, BigDecimal, ZonedDateTime, BigDecimal, BigDecimal,
                Void, Void, Void, Void> soVars = computeSwitchOutVars(req);
        BigDecimal soAmount = soVars.getSecond();
        BigDecimal soUnits = soVars.getThird();
        BigDecimal soNav = soVars.getFifth();
        BigDecimal soAmountDet = soVars.getSixth();

        Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> purchAmt =
                computePurchaseAmounts(req, nav, newPurchaseTypeFlag, aggregPlusTxForLimitCheck,
                        soAmount, soAmountDet);
        BigDecimal sumPurchaseAmtCdt = purchAmt.getFirst();
        BigDecimal sumPurchaseAmtDet = purchAmt.getSecond();
        BigDecimal currentPurchaseAmt = computeCurrentPurchaseAmount(req, nav, soAmount, soUnits, soNav);

        boolean investorPlanExists = computeInvestorPlanExists(req);

        Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String> limitVals = fetchLimitValues(req, subTrxnType, nav);
        Decuple<String, BigDecimal, BigDecimal, BigDecimal, Boolean, Boolean,
                Boolean, Void, Void, Void> limitMeta = fetchLimitMetadata(req, subTrxnType, nav);

        boolean limitsFound = Boolean.TRUE.equals(limitVals.getFirst());
        if (!limitsFound) return result;

        BigDecimal minValue = limitVals.getSecond();
        BigDecimal maxValue = limitVals.getThird();
        BigDecimal newMinValue = limitVals.getFourth();
        BigDecimal newMaxValue = limitVals.getFifth();
        BigDecimal multiples = limitVals.getSixth();
        String schFundSubfund = limitVals.getTenth();

        String includeCurrentBalance = limitMeta.getFirst();
        BigDecimal aggrAmount = limitMeta.getThird();
        boolean tooManySchemeRows = Boolean.TRUE.equals(limitMeta.getFifth());
        boolean tooManySubFundRows = Boolean.TRUE.equals(limitMeta.getSixth());
        boolean tooManyFundRows = Boolean.TRUE.equals(limitMeta.getSeventh());

        String checkLimitPlus = getCheckLimitPlus(req);

        String schemeCode = req.getContext().getScheme() != null
                ? req.getContext().getScheme().getSchemeCode() : null;
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        List<SchemeLimits> schemeLevel = filterBySchemeCode(slList, schemeCode);
        SchemeLimits sl = (schemeLevel.size() == 1) ? schemeLevel.get(0) : null;

        String otherAmcParam = getApplicationParameter(req, "OTHER_AMC");
        String allowCartingSifFlag = getApplicationParameter(req, "ALLOW_CARTING_ONLINE_SIF");

        if (sl != null && nvl(sl.getAccreditedInvLimit()).compareTo(ZERO) > 0) {
            if ("Y".equals(req.getContext().getAccreditedInvFlag())) {
                minValue = nvl(sl.getAccreditedInvLimit());
                newMinValue = minValue;
            }
        }

        String amcCode = getAmcCode(req);
        if ("L".equals(amcCode) && sl != null
                && nvl(sl.getBrokerInvestLimit()).compareTo(ZERO) > 0) {
            if (req.getContext().getTransactionType() == TransactionType.PURCHASE) {
                minValue = nvl(sl.getBrokerInvestLimit());
                newMinValue = minValue;
            }
        }

        if ("Y".equals(amcCode) && req.getContext().getTransactionType() == TransactionType.PURCHASE) {
            if (req.getProspect() != null) {
                boolean cdtExistsForFolio = cdtp != null;
                if (!cdtExistsForFolio) {
                    newPurchaseTypeFlag = "Y";
                }
            }
        }

        if ("Y".equals(req.getContext().getAccreditedInvFlag())
                && "SIF".equals(otherAmcParam)
                && getSifFolioExistsCount(req) == 0
                && sl != null && nvl(sl.getAccreditedInvLimit()).compareTo(ZERO) > 0) {
            minValue = nvl(sl.getAccreditedInvLimit());
            newMinValue = nvl(sl.getAccreditedInvLimit());
        }

        boolean skipStandardMinimumCheck = false;
        if ("Y".equals(allowCartingSifFlag) && "SIF".equals(otherAmcParam)) {
            String cartRefNo = cdtp != null ? cdtp.getCartRefNo() : null;
            if (cartRefNo != null && !cartRefNo.isEmpty() && "Y".equals(onlineUserFlag)) {
                skipStandardMinimumCheck = true;
            }
        }

        // Excel 2.3: AMC '02' Tax Status / Occupation Limits override
        if ("02".equals(amcCode)
                && "P".equals(RuleUnitUtils.deriveDbCrType(req))
                && schemeCode != null && ("01".equals(schemeCode) || "02".equals(schemeCode))) {
            TaxStatusOccLimits taxOccLimits = req.getTaxStatusOccLimits();
            if (taxOccLimits != null) {
                if ("01".equals(schemeCode)) {
                    if (taxOccLimits.getNewPurMinValue() != null) newMinValue = taxOccLimits.getNewPurMinValue();
                    if (taxOccLimits.getNewPurMaxValue() != null) newMaxValue = taxOccLimits.getNewPurMaxValue();
                } else {
                    if (taxOccLimits.getAddlPurMinValue() != null) minValue = taxOccLimits.getAddlPurMinValue();
                    if (taxOccLimits.getAddlPurMaxValue() != null) maxValue = taxOccLimits.getAddlPurMaxValue();
                }
            }
        }

        if (!investorPlanExists || "Y".equals(newPurchaseTypeFlag)) {
            maxValue = nvl(newMaxValue);
            minValue = nvl(newMinValue);
        }

        if ("Y".equals(includeCurrentBalance)) {
            minValue = nvl(newMinValue);
            sumPurchaseAmtDet = nvl(aggrAmount);
        }

        result[0] = Boolean.TRUE;
        result[10] = tooManySchemeRows;
        result[11] = tooManySubFundRows;
        result[12] = tooManyFundRows;
        result[13] = req.getSchemeSetup() != null ? nvlStr(req.getSchemeSetup().getSchemeCode(), "") : "";
        result[14] = req.getSchemeSetup() != null ? nvlStr(req.getSchemeSetup().getParentSubFundCode(), "") : "";
        result[15] = req.getSchemeSetup() != null ? nvlStr(req.getSchemeSetup().getParentFundCode(), "") : "";
        result[16] = req.getContext().getTransactionType() != null ? req.getContext().getTransactionType().name() : "";

        String trxnSign = getTransactionSign(req);
        boolean signGuardPassed = "+".equals(trxnSign)
                && !"R".equals(req.getContext().getTransactionMode())
                && unitsAmountFlag != null
                && ("U".equals(unitsAmountFlag) || "A".equals(unitsAmountFlag))
                && "Y".equals(checkLimitPlus);
        result[1] = signGuardPassed;

        BigDecimal totalPurchaseAmt;
        if ("S".equals(schFundSubfund)) {
            totalPurchaseAmt = nvl(sumPurchaseAmtCdt).add(nvl(sumPurchaseAmtDet));
        } else {
            totalPurchaseAmt = nvl(sumPurchaseAmtCdt).add(nvl(aggrAmount));
        }
        result[4] = totalPurchaseAmt;
        result[3] = nvl(maxValue);
        result[2] = nvl(minValue);
        result[6] = nvl(multiples);

        if (!skipStandardMinimumCheck) {
            if (swapOption != null) {
                if ("T".equals(swapOption) && "L".equals(amcCode)) skipStandardMinimumCheck = true;
                if ("Y".equals(swapOption) || "Q".equals(swapOption)) skipStandardMinimumCheck = true;
            }
        }
        result[8] = skipStandardMinimumCheck;

        BigDecimal minAmountTotal;
        if ("S".equals(schFundSubfund)) {
            if ("Y".equals(includeCurrentBalance)) {
                minAmountTotal = nvl(currentPurchaseAmt).add(nvl(sumPurchaseAmtDet));
            } else {
                minAmountTotal = nvl(currentPurchaseAmt);
            }
        } else {
            minAmountTotal = nvl(currentPurchaseAmt).add(nvl(aggrAmount));
        }
        result[5] = minAmountTotal;

        boolean skipMultiplesCheck = isElssExemptFromMultiples(req);
        if (!skipMultiplesCheck) {
            if (nvl(multiples).compareTo(ZERO) <= 0) skipMultiplesCheck = true;
            else if ("Y".equals(swapOption)) skipMultiplesCheck = true;
            else if (spouseFirstName != null && "Transferred from ceased EBT".equals(spouseFirstName))
                skipMultiplesCheck = true;
        }
        result[9] = skipMultiplesCheck;

        int scale = getDecimalAmount(req);
        result[7] = nvl(currentPurchaseAmt).setScale(scale, RoundingMode.HALF_UP);

        return result;
    }

    public static boolean isPurchaseAmtBaseGuardPassed(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[0]);
    }
    public static boolean isPurchaseAmtSignGuardPassed(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[1]);
    }
    public static BigDecimal getPurchaseAmtEffectiveMin(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[2];
    }
    public static BigDecimal getPurchaseAmtEffectiveMax(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[3];
    }
    public static BigDecimal getPurchaseAmtTotal(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[4];
    }
    public static BigDecimal getPurchaseAmtMinTotal(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[5];
    }
    public static BigDecimal getPurchaseAmtMultiples(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[6];
    }
    public static BigDecimal getPurchaseAmtCurrentRounded(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseAmountInternal(req)[7];
    }
    public static boolean isPurchaseAmtSkipMinCheck(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[8]);
    }
    public static boolean isPurchaseAmtSkipMultiplesCheck(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[9]);
    }
    public static boolean isPurchaseAmtTooManySchemeRows(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[10]);
    }
    public static boolean isPurchaseAmtTooManySubFundRows(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[11]);
    }
    public static boolean isPurchaseAmtTooManyFundRows(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseAmountInternal(req)[12]);
    }
    public static String getPurchaseAmtSchemeCode(RuleEngineRequest req) {
        return (String) computePurchaseAmountInternal(req)[13];
    }
    public static String getPurchaseAmtParentSubFundCode(RuleEngineRequest req) {
        return (String) computePurchaseAmountInternal(req)[14];
    }
    public static String getPurchaseAmtParentFundCode(RuleEngineRequest req) {
        return (String) computePurchaseAmountInternal(req)[15];
    }
    public static String getPurchaseAmtTxnTypeName(RuleEngineRequest req) {
        return (String) computePurchaseAmountInternal(req)[16];
    }

    private static Object[] computeRedemptionAmountInternal(RuleEngineRequest req) {
        // [0] signGuardPassed, [1] effectiveMax, [2] effectiveMin, [3] totalRedemptionAmount,
        // [4] currentRedemptionAmount, [5] multiples, [6] currentRedemptionAmountRounded,
        // [7] skipMinCheck, [8] skipMultiplesCheck
        Object[] result = new Object[] {
                Boolean.FALSE, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO,
                Boolean.FALSE, Boolean.FALSE
        };

        if (req == null || req.getContext() == null) return result;

        BigDecimal nav = getNav(req);
        Long autoTrxnNo = computeAutoTrxnNo(req);
        String unitsAmountFlag = computeUnitsAmountFlag(req, nav);
        String subTrxnType = computeSubTrxnType(req);

        Decuple<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void> swapFreq = computeSwapAndFrequency(req, autoTrxnNo);
        String swapOption = swapFreq.getFirst();

        ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
        String sourceProgram = cdtp != null ? cdtp.getSourceProgram() : null;
        String motherName = cdtp != null ? cdtp.getMotherName() : null;
        String spouseFirstName = cdtp != null ? cdtp.getSpouseFirstName() : null;
        String sipFlag = req.getContext().getSipFlag();

        String userType = getUserType(req);
        String dpIdRequired = getDpIdRequired(req);

        String applyLimits = computeApplyLimits(req, autoTrxnNo, swapOption, sourceProgram,
                userType, dpIdRequired, motherName);
        if ("X".equals(applyLimits)) return result;

        Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> redemptionAmt =
                computeRedemptionAmounts(req);
        BigDecimal sumRedemptionAmtCdt = redemptionAmt.getFirst();
        BigDecimal sumRedemptionAmtDet = redemptionAmt.getSecond();
        BigDecimal currentRedemptionAmt = computeCurrentRedemptionAmount(req);

        Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String> limitVals = fetchLimitValues(req, subTrxnType, nav);

        boolean limitsFound = Boolean.TRUE.equals(limitVals.getFirst());
        if (!limitsFound) return result;

        BigDecimal minValue = limitVals.getSecond();
        BigDecimal maxValue = limitVals.getThird();
        BigDecimal multiples = limitVals.getSixth();

        String schemeCode = req.getContext().getScheme() != null
                ? req.getContext().getScheme().getSchemeCode() : null;
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        List<SchemeLimits> schemeLevel = filterBySchemeCode(slList, schemeCode);
        SchemeLimits sl = (schemeLevel.size() == 1) ? schemeLevel.get(0) : null;

        if (sl != null && nvl(sl.getAccreditedInvLimit()).compareTo(ZERO) > 0) {
            if ("Y".equals(req.getContext().getAccreditedInvFlag())) {
                minValue = nvl(sl.getAccreditedInvLimit());
            }
        }

        // baseGuardPassed is not in the output array - it was only in the old Result class
        // The caller uses signGuardPassed instead

        String checkLimitMinus = getCheckLimitMinus(req);
        String trxnSign = getTransactionSign(req);
        boolean signGuardPassed = "-".equals(trxnSign)
                && !"Y".equals(sipFlag)
                && "A".equals(unitsAmountFlag)
                && "Y".equals(checkLimitMinus);
        result[0] = signGuardPassed;

        BigDecimal totalRedeem = nvl(sumRedemptionAmtCdt).add(nvl(sumRedemptionAmtDet));
        result[3] = totalRedeem;
        result[1] = nvl(maxValue);
        result[2] = nvl(minValue);
        result[5] = nvl(multiples);
        result[4] = nvl(currentRedemptionAmt);

        boolean skipMinCheck = false;
        String amcCode = getAmcCode(req);
        if (("L".equals(amcCode) && "T".equals(swapOption))
                || ("H".equals(amcCode) && "MB".equals(req.getContext().getSubTransactionType()))) {
            skipMinCheck = true;
        }
        if ("R".equals(RuleUnitUtils.deriveDbCrType(req)) && "EXG".equals(userType)
                && "Y".equals(dpIdRequired)) {
            String excludeFlag = getApplicationParameter(req, "EXCLUDE_MIN_AMOUNT_UNIT_VALIDATION_EXG");
            if ("Y".equals(excludeFlag)) skipMinCheck = true;
        }
        if (swapOption != null && ("Y".equals(swapOption) || "Q".equals(swapOption))) {
            skipMinCheck = true;
        }
        result[7] = skipMinCheck;

        boolean skipMultiplesCheck = false;
        if (nvl(multiples).compareTo(ZERO) <= 0) skipMultiplesCheck = true;
        else if ("Y".equals(swapOption)) skipMultiplesCheck = true;
        else if (spouseFirstName != null && "Transferred from ceased EBT".equals(spouseFirstName))
            skipMultiplesCheck = true;
        else if ("H".equals(amcCode) && "MB".equals(req.getContext().getSubTransactionType()))
            skipMultiplesCheck = true;
        result[8] = skipMultiplesCheck;

        int scale = getDecimalAmount(req);
        result[6] = nvl(currentRedemptionAmt).setScale(scale, RoundingMode.HALF_UP);

        return result;
    }

    public static boolean isRedemptionAmtSignGuardPassed(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeRedemptionAmountInternal(req)[0]);
    }
    public static BigDecimal getRedemptionAmtEffectiveMax(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[1];
    }
    public static BigDecimal getRedemptionAmtEffectiveMin(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[2];
    }
    public static BigDecimal getRedemptionAmtTotal(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[3];
    }
    public static BigDecimal getRedemptionAmtCurrent(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[4];
    }
    public static BigDecimal getRedemptionAmtMultiples(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[5];
    }
    public static BigDecimal getRedemptionAmtCurrentRounded(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionAmountInternal(req)[6];
    }
    public static boolean isRedemptionAmtSkipMinCheck(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeRedemptionAmountInternal(req)[7]);
    }
    public static boolean isRedemptionAmtSkipMultiplesCheck(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeRedemptionAmountInternal(req)[8]);
    }

    private static Object[] computePurchaseUnitInternal(RuleEngineRequest req) {
        // [0] signGuardPassed, [1] maxUnits, [2] maxPerc, [3] minUnits,
        // [4] totalPurchaseUnits, [5] totalPurchaseUnitsForMin, [6] closingBalance, [7] aggrClosingBalance
        Object[] result = new Object[] {
                Boolean.FALSE, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO
        };

        if (req == null || req.getContext() == null) return result;

        BigDecimal nav = getNav(req);
        Long autoTrxnNo = computeAutoTrxnNo(req);
        String unitsAmountFlag = computeUnitsAmountFlag(req, nav);
        String subTrxnType = computeSubTrxnType(req);
        String newPurchaseTypeFlag = computeNewPurchaseTypeFlag(req);

        Decuple<String, String, String, Void, Void, Void, Void, Void, Void, Void> aggreg = computeAggregFlags(req);
        String aggregPlusTxForLimitCheck = aggreg.getSecond();

        Decuple<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void> swapFreq = computeSwapAndFrequency(req, autoTrxnNo);
        String swapOption = swapFreq.getFirst();

        ConsolidatedDailyTradesPrimary cdt = req.getConsolidatedDailyTradesPrimary();
        String sourceProgram = cdt != null ? cdt.getSourceProgram() : null;
        String motherName = cdt != null ? cdt.getMotherName() : null;

        String userType = getUserType(req);
        String dpIdRequired = getDpIdRequired(req);

        String applyLimits = computeApplyLimits(req, autoTrxnNo, swapOption, sourceProgram,
                userType, dpIdRequired, motherName);
        if ("X".equals(applyLimits)) return result;

        Decuple<String, BigDecimal, BigDecimal, ZonedDateTime, BigDecimal, BigDecimal,
                Void, Void, Void, Void> soVars = computeSwitchOutVars(req);
        BigDecimal soAmount = soVars.getSecond();
        BigDecimal soUnits = soVars.getThird();
        BigDecimal soNav = soVars.getFifth();

        Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> purchUnits =
                computePurchaseUnits(req, nav, newPurchaseTypeFlag, aggregPlusTxForLimitCheck,
                        soUnits, soNav, soAmount);
        BigDecimal sumPurchaseUnitsCdt = purchUnits.getFirst();
        BigDecimal sumPurchaseUnitsDet = purchUnits.getSecond();
        BigDecimal currentPurchaseUnits = computeCurrentPurchaseUnits(req, nav, soAmount, soUnits, soNav);

        Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String> limitVals = fetchLimitValues(req, subTrxnType, nav);
        Decuple<String, BigDecimal, BigDecimal, BigDecimal, Boolean, Boolean,
                Boolean, Void, Void, Void> limitMeta = fetchLimitMetadata(req, subTrxnType, nav);

        boolean limitsFound = Boolean.TRUE.equals(limitVals.getFirst());
        if (!limitsFound) return result;

        BigDecimal minUnits = limitVals.getSeventh();
        BigDecimal maxUnits = limitVals.getEighth();
        BigDecimal maxPerc = limitVals.getNinth();
        String schFundSubfund = limitVals.getTenth();

        String includeCurrentBalance = limitMeta.getFirst();
        BigDecimal aggrUnits = limitMeta.getSecond();
        BigDecimal aggrClosingBal = limitMeta.getFourth();

        if ("Y".equals(includeCurrentBalance)) {
            sumPurchaseUnitsDet = nvl(aggrUnits);
        }

        String checkLimitPlus = getCheckLimitPlus(req);
        BigDecimal schemeControlsClosingBalance = getSchemeControlsClosingBalance(req);

        String trxnSign = getTransactionSign(req);
        String trxnMode = req.getContext().getTransactionMode();

        boolean signGuardPassed = "+".equals(trxnSign)
                && !"R".equals(trxnMode)
                && "U".equals(unitsAmountFlag)
                && "Y".equals(checkLimitPlus);

        result[0] = signGuardPassed;
        result[1] = nvl(maxUnits);
        result[2] = nvl(maxPerc);
        result[3] = nvl(minUnits);
        if ("S".equals(schFundSubfund)) {
            result[6] = nvl(schemeControlsClosingBalance);
            result[7] = ZERO;
        } else {
            result[6] = ZERO;
            result[7] = nvl(aggrClosingBal);
        }

        BigDecimal totalUnits;
        if ("S".equals(schFundSubfund)) {
            totalUnits = nvl(sumPurchaseUnitsCdt).add(nvl(sumPurchaseUnitsDet));
        } else {
            totalUnits = nvl(sumPurchaseUnitsCdt).add(nvl(aggrUnits));
        }
        result[4] = totalUnits;

        BigDecimal totalForMin;
        if ("S".equals(schFundSubfund)) {
            if ("Y".equals(includeCurrentBalance)) {
                totalForMin = nvl(currentPurchaseUnits).add(nvl(sumPurchaseUnitsDet));
            } else {
                totalForMin = nvl(currentPurchaseUnits);
            }
        } else {
            totalForMin = nvl(currentPurchaseUnits).add(nvl(aggrUnits));
        }
        result[5] = totalForMin;

        return result;
    }

    public static boolean isPurchaseUnitSignGuardPassed(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computePurchaseUnitInternal(req)[0]);
    }
    public static BigDecimal getPurchaseUnitMaxUnits(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[1];
    }
    public static BigDecimal getPurchaseUnitMaxPerc(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[2];
    }
    public static BigDecimal getPurchaseUnitMinUnits(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[3];
    }
    public static BigDecimal getPurchaseUnitTotal(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[4];
    }
    public static BigDecimal getPurchaseUnitTotalForMin(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[5];
    }
    public static BigDecimal getPurchaseUnitClosingBalance(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[6];
    }
    public static BigDecimal getPurchaseUnitAggrClosingBalance(RuleEngineRequest req) {
        return (BigDecimal) computePurchaseUnitInternal(req)[7];
    }
    public static BigDecimal getPurchaseUnitPercentage(RuleEngineRequest req) {
        BigDecimal closingBal = getPurchaseUnitClosingBalance(req);
        if (closingBal.compareTo(ZERO) <= 0) return ZERO;
        return getPurchaseUnitTotal(req)
                .divide(closingBal, 10, RoundingMode.HALF_UP)
                .multiply(HUNDRED);
    }
    public static BigDecimal getPurchaseUnitAggrPercentage(RuleEngineRequest req) {
        BigDecimal closingBal = getPurchaseUnitAggrClosingBalance(req);
        if (closingBal.compareTo(ZERO) <= 0) return ZERO;
        return getPurchaseUnitTotal(req)
                .divide(closingBal, 10, RoundingMode.HALF_UP)
                .multiply(HUNDRED);
    }

    private static Object[] computeRedemptionUnitInternal(RuleEngineRequest req) {
        // [0] signGuardPassed, [1] maxUnits, [2] maxPerc, [3] minUnits, [4] minValue,
        // [5] totalRedemptionUnits, [6] currentRedemptionUnits, [7] closingBalance,
        // [8] nav, [9] shouldRaiseMinUnitsAmtError
        Object[] result = new Object[] {
                Boolean.FALSE, ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO,
                ZERO, Boolean.FALSE
        };

        if (req == null || req.getContext() == null) return result;

        BigDecimal nav = getNav(req);
        Long autoTrxnNo = computeAutoTrxnNo(req);
        String unitsAmountFlag = computeUnitsAmountFlag(req, nav);
        String subTrxnType = computeSubTrxnType(req);

        Decuple<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void> swapFreq = computeSwapAndFrequency(req, autoTrxnNo);
        String swapOption = swapFreq.getFirst();

        ConsolidatedDailyTradesPrimary cdt = req.getConsolidatedDailyTradesPrimary();
        String sourceProgram = cdt != null ? cdt.getSourceProgram() : null;
        String motherName = cdt != null ? cdt.getMotherName() : null;
        String sipFlag = req.getContext().getSipFlag();

        String userType = getUserType(req);
        String dpIdRequired = getDpIdRequired(req);

        String applyLimits = computeApplyLimits(req, autoTrxnNo, swapOption, sourceProgram,
                userType, dpIdRequired, motherName);
        if ("X".equals(applyLimits)) return result;

        Decuple<BigDecimal, BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void> redemptionUnits =
                computeRedemptionUnits(req);
        BigDecimal sumRedemptionUnitsCdt = redemptionUnits.getFirst();
        BigDecimal sumRedemptionUnitsDet = redemptionUnits.getSecond();
        BigDecimal currentRedemptionUnits = redemptionUnits.getThird();

        Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String> limitVals = fetchLimitValues(req, subTrxnType, nav);

        boolean limitsFound = Boolean.TRUE.equals(limitVals.getFirst());
        if (!limitsFound) return result;

        BigDecimal minValue = limitVals.getSecond();
        BigDecimal minUnits = limitVals.getSeventh();
        BigDecimal maxUnits = limitVals.getEighth();
        BigDecimal maxPerc = limitVals.getNinth();

        String schemeCode = req.getContext().getScheme() != null
                ? req.getContext().getScheme().getSchemeCode() : null;
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        List<SchemeLimits> schemeLevel = filterBySchemeCode(slList, schemeCode);
        SchemeLimits sl = (schemeLevel.size() == 1) ? schemeLevel.get(0) : null;

        if (sl != null && nvl(sl.getAccreditedInvLimit()).compareTo(ZERO) > 0) {
            if ("Y".equals(req.getContext().getAccreditedInvFlag())) {
                minValue = nvl(sl.getAccreditedInvLimit());
            }
        }

        String checkLimitMinus = getCheckLimitMinus(req);
        BigDecimal schemeControlsClosingBalance = getSchemeControlsClosingBalance(req);
        BigDecimal folioClosingBalance = getFolioClosingBalance(req);

        String trxnSign = getTransactionSign(req);
        boolean signGuardPassed = "-".equals(trxnSign)
                && !"Y".equals(sipFlag)
                && "U".equals(unitsAmountFlag)
                && "Y".equals(checkLimitMinus);

        result[0] = signGuardPassed;
        result[1] = nvl(maxUnits);
        result[2] = nvl(maxPerc);
        result[3] = nvl(minUnits);
        result[4] = nvl(minValue);
        result[5] = nvl(sumRedemptionUnitsCdt).add(nvl(sumRedemptionUnitsDet));
        result[6] = nvl(currentRedemptionUnits);
        result[7] = nvl(schemeControlsClosingBalance);
        result[8] = nvl(nav);

        String trxnMode = req.getContext().getTransactionMode();
        String amcCode = getAmcCode(req);
        boolean skipMinCheck = false;

        if ("H".equals(amcCode) && "MB".equals(req.getContext().getSubTransactionType())) {
            skipMinCheck = true;
        }
        if ("EXG".equals(userType) && "Y".equals(dpIdRequired)) {
            String excludeFlag = getApplicationParameter(req, "EXCLUDE_MIN_AMOUNT_UNIT_VALIDATION_EXG");
            if ("Y".equals(excludeFlag)) skipMinCheck = true;
        }

        boolean shouldRaise = false;
        if (!skipMinCheck && !"R".equals(trxnMode)
                && nvl(currentRedemptionUnits).compareTo(nvl(minUnits)) < 0) {
            String dbCr = RuleUnitUtils.deriveDbCrType(req);
            if ("R".equals(dbCr) || "SO".equals(dbCr)) {
                shouldRaise = true;
                if (("TH".equals(amcCode) || "H".equals(amcCode))
                        && "CAMSWEB".equals(req.getContext().getUserCode())) {
                    if (nvl(sumRedemptionUnitsCdt).compareTo(nvl(folioClosingBalance)) >= 0) {
                        shouldRaise = false;
                    }
                }
                if (shouldRaise) {
                    BigDecimal amtCheck = nvl(currentRedemptionUnits).multiply(nvl(nav));
                    if (amtCheck.compareTo(nvl(minValue)) >= 0) {
                        shouldRaise = false;
                    }
                }
            }
        }
        result[9] = shouldRaise;
        return result;
    }

    public static boolean isRedemptionUnitSignGuardPassed(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeRedemptionUnitInternal(req)[0]);
    }
    public static BigDecimal getRedemptionUnitMaxUnits(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[1];
    }
    public static BigDecimal getRedemptionUnitMaxPerc(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[2];
    }
    public static BigDecimal getRedemptionUnitMinUnits(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[3];
    }
    public static BigDecimal getRedemptionUnitMinValue(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[4];
    }
    public static BigDecimal getRedemptionUnitTotal(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[5];
    }
    public static BigDecimal getRedemptionUnitCurrent(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[6];
    }
    public static BigDecimal getRedemptionUnitClosingBalance(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[7];
    }
    public static BigDecimal getRedemptionUnitNav(RuleEngineRequest req) {
        return (BigDecimal) computeRedemptionUnitInternal(req)[8];
    }
    public static boolean isRedemptionUnitShouldRaiseMinError(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeRedemptionUnitInternal(req)[9]);
    }
    public static BigDecimal getRedemptionUnitCurrentAmountCheck(RuleEngineRequest req) {
        try {
            return getRedemptionUnitCurrent(req).multiply(getRedemptionUnitNav(req));
        } catch (Exception e) {
            return ZERO;
        }
    }

    public static BigDecimal getRedemptionUnitPercentage(RuleEngineRequest req) {
        BigDecimal closingBal = getRedemptionUnitClosingBalance(req);
        if (closingBal.compareTo(ZERO) <= 0) return ZERO;
        return getRedemptionUnitTotal(req)
                .divide(closingBal, 10, RoundingMode.HALF_UP)
                .multiply(HUNDRED);
    }

    private static Object[] computeMaxInvestmentInternal(RuleEngineRequest req) {
        // [0] investorLimitApplies, [1] totalCostExceedsLimit, [2] hasExceptionConfig,
        // [3] withinExceptionLimit, [4] totalCostValue, [5] investorMaxLimit,
        // [6] corpusPercentLimitValue, [7] exceptionTxnAmountLimit,
        // [8] panIntraApplies, [9] panIntraExceeded, [10] panAggAmtForEntryDate,
        // [11] panMaxLimit, [12] entryDateStr, [13] panDateApplies, [14] tradeDateMode,
        // [15] patWindowCheckRequired, [16] panAggAmtForTradeDate, [17] panAggAmtForEntryDate1,
        // [18] tradeDateStr, [19] panMonthlyApplies, [20] skipMonthlyForTxnType,
        // [21] panAggAmtForMonthWindow, [22] monthLabel
        Object[] result = new Object[] {
                Boolean.FALSE, Boolean.FALSE, Boolean.FALSE, Boolean.FALSE,
                ZERO, ZERO, ZERO, ZERO,
                Boolean.FALSE, Boolean.FALSE, ZERO, ZERO, "",
                Boolean.FALSE, Boolean.FALSE, Boolean.FALSE, ZERO, ZERO, "",
                Boolean.FALSE, Boolean.FALSE, ZERO, ""
        };

        if (req == null || req.getContext() == null) return result;

        SubFundSetup sf = req.getSubFundSetup();
        if (sf == null) return result;

        RuleEngineRequestContext ctx = req.getContext();
        TransactionType txnType = ctx.getTransactionType();
        if (txnType == null) return result;

        boolean isPurchase = txnType == TransactionType.PURCHASE;
        boolean isSwitchOut = txnType == TransactionType.SWITCH_OUT;
        boolean isSwitchIn = txnType == TransactionType.SWITCH_IN;
        boolean isSwitch = isSwitchOut || isSwitchIn;

        BigDecimal investorMaxLimitRaw = nvl(sf.getInvestorInvestmentLimit());
        ZonedDateTime investorLimitStartDate = sf.getInvInvestLmtAddDt();
        String investorAggregationMode = sf.getInvInvestLmtQuarterFlag();
        String aggregateAtPan = nvlStr(sf.getAggregateAtPanLevel(), "N");
        BigDecimal panMaxLimitRaw = nvl(sf.getPanLevelMaxInvAmountLimit());
        BigDecimal exceptionTxnAmountLimit = nvl(sf.getExceptionTrxnAmountLimit());
        BigDecimal corpusPercentForExceptions = nvl(sf.getCorpusPercForExcepTrxnAmt());
        ZonedDateTime panAggSysRegnFrom = sf.getAggPanForSysRegnFrom();
        String sfAmcCode = nvlStr(sf.getAmcCode(), "#$");
        String panLevelSetupEnabled = nvlStr(sf.getPanLevelMaxInvestmentAmountSetupFlag(), "N");
        String addlInvestorLimitFlag = nvlStr(sf.getAddlInvestorLimitValidation(), "N");
        BigDecimal maxInvestorLimitValue = nvl(sf.getMaxInvestorLimitValue());
        BigDecimal maxAggregateLimitPercent = nvl(sf.getMaxAggregateLimitPercentage());
        String schemeAumPeriod = sf.getSchemeAumPeriod();
        String intraSchemePolicy = sf.getIntraSchemeSwitchLimit();
        BigDecimal intraSchemeSwitchAmtLimit = nvl(sf.getIntraSchemeSwitchAmtLimit());
        ZonedDateTime intraSchemeSwitchLimitUpto = sf.getIntraSchemeSwitchLimitUpto();
        String monthlyPanAggType = sf.getPanLevelMaxInvAggrType();

        ZonedDateTime tradeDate = ctx.getTradeDate();
        ZonedDateTime entryDate = ctx.getEntryDate();
        String schemeCode = ctx.getScheme() != null ? ctx.getScheme().getSchemeCode() : null;
        BigDecimal transactionAmount = ctx.getAmount() != null ? ctx.getAmount().getAmountValue() : null;
        BigDecimal transactionUnits = ctx.getUnits();
        ZonedDateTime patRegistrationDate = ctx.getRegistrationDate();
        String isInterSwitchFlag = getInterSwitchFlag(req);

        BigDecimal newAmount;
        if (!isSwitchIn) {
            if (transactionAmount == null || transactionAmount.compareTo(ZERO) == 0) {
                BigDecimal navForCalc = getNavForCalc(req, schemeCode, tradeDate);
                newAmount = nvl(transactionUnits).multiply(navForCalc);
            } else {
                newAmount = transactionAmount;
            }
        } else {
            if (transactionAmount == null || transactionAmount.compareTo(ZERO) == 0) {
                BigDecimal navForCalc = getNavForCalc(req, schemeCode, tradeDate);
                newAmount = nvl(transactionUnits).multiply(navForCalc);
            } else {
                newAmount = transactionAmount;
            }
        }

        BigDecimal effectiveTxnAmount = nvl(newAmount);

        BigDecimal investorMaxLimit = investorMaxLimitRaw;
        if ("Y".equals(addlInvestorLimitFlag)) {
            if (maxInvestorLimitValue.compareTo(ZERO) > 0
                    && maxInvestorLimitValue.compareTo(investorMaxLimit) > 0) {
                investorMaxLimit = maxInvestorLimitValue;
            } else if (maxAggregateLimitPercent.compareTo(ZERO) > 0
                    && schemeAumPeriod != null && schemeCode != null) {
                BigDecimal schemeAumValue = nvl(sf.getSchemeAumValue());
                investorMaxLimit = schemeAumValue.multiply(maxAggregateLimitPercent)
                        .divide(HUNDRED, 2, RoundingMode.HALF_UP);
            }
        }

        BigDecimal costValueAggregated = ZERO;
        if (investorMaxLimit.compareTo(ZERO) > 0
                && tradeDate != null && investorLimitStartDate != null
                && !tradeDate.isBefore(investorLimitStartDate)) {
            if ("I".equals(investorAggregationMode)) {
                costValueAggregated = nvl(sf.getCalcCostValueAgg());
            }
        }
        BigDecimal totalCostValue = costValueAggregated.add(effectiveTxnAmount);

        BigDecimal corpusPercentLimitValue = ZERO;
        if (totalCostValue.compareTo(investorMaxLimit) >= 0
                && exceptionTxnAmountLimit.compareTo(ZERO) > 0
                && corpusPercentForExceptions.compareTo(ZERO) > 0
                && schemeCode != null) {
            BigDecimal prevNavVal = nvl(sf.getPrevNavValue());
            BigDecimal schemeBal = nvl(sf.getSchemeBalanceAtPrevNav());
            corpusPercentLimitValue = prevNavVal.multiply(schemeBal)
                    .multiply(corpusPercentForExceptions)
                    .divide(HUNDRED, 2, RoundingMode.HALF_UP);
        }

        BigDecimal panMaxLimit = panMaxLimitRaw;
        if (isSwitch && "N".equals(isInterSwitchFlag)
                && "Y".equals(aggregateAtPan) && intraSchemePolicy != null) {
            String policy = nvlStr(intraSchemePolicy, "EPL");
            if (("EPL".equals(policy) && panMaxLimitRaw.compareTo(ZERO) > 0)
                    || ("NISL".equals(policy) && intraSchemeSwitchLimitUpto != null
                        && tradeDate != null && tradeDate.isAfter(intraSchemeSwitchLimitUpto))
                    || ("NL".equals(policy) && intraSchemeSwitchLimitUpto != null
                        && tradeDate != null && tradeDate.isAfter(intraSchemeSwitchLimitUpto))) {
                panMaxLimit = panMaxLimitRaw;
            } else if ("NISL".equals(policy)
                    && intraSchemeSwitchAmtLimit.compareTo(panMaxLimitRaw) > 0
                    && (intraSchemeSwitchLimitUpto == null
                        || (tradeDate != null && !tradeDate.isAfter(intraSchemeSwitchLimitUpto)))) {
                panMaxLimit = intraSchemeSwitchAmtLimit;
            } else if ("NL".equals(policy)
                    && intraSchemeSwitchLimitUpto != null
                    && tradeDate != null && !tradeDate.isAfter(intraSchemeSwitchLimitUpto)) {
                panMaxLimit = ZERO;
            }
        }

        PanLevelMaxInvAmtSetup panSetup = req.getPanLevelMaxInvAmtSetup();
        ZonedDateTime panAggSysRegnTo = ZonedDateTime.of(2999, 12, 31, 0, 0, 0, 0,
                tradeDate != null ? tradeDate.getZone() : java.time.ZoneOffset.UTC);
        if ("Y".equals(panLevelSetupEnabled)
                && patRegistrationDate != null
                && panSetup != null
                && nvl(panSetup.getPanLevelMaxInvAmount()).compareTo(ZERO) > 0
                && panSetup.getFromDate() != null) {
            panMaxLimit = panSetup.getPanLevelMaxInvAmount();
            panAggSysRegnFrom = panSetup.getFromDate();
            panAggSysRegnTo = panSetup.getToDate();
        }

        // Investor-Level Check (3830017/18/19)
        boolean investorLimitApplies = investorMaxLimit.compareTo(ZERO) > 0
                && tradeDate != null && investorLimitStartDate != null
                && !tradeDate.isBefore(investorLimitStartDate)
                && totalCostValue.compareTo(ZERO) > 0;
        boolean totalCostExceedsLimit = totalCostValue.compareTo(investorMaxLimit) >= 0;
        boolean hasExceptionConfig = exceptionTxnAmountLimit.compareTo(ZERO) > 0
                && corpusPercentForExceptions.compareTo(ZERO) > 0;
        boolean withinExceptionLimit = totalCostValue.compareTo(exceptionTxnAmountLimit) < 0;

        result[0] = investorLimitApplies;
        result[1] = totalCostExceedsLimit;
        result[2] = hasExceptionConfig;
        result[3] = withinExceptionLimit;
        result[4] = totalCostValue;
        result[5] = investorMaxLimit;
        result[6] = corpusPercentLimitValue;
        result[7] = exceptionTxnAmountLimit;

        // PAN Intra-Scheme Switch (3830020)
        boolean panIntraApplies = isSwitch && "N".equals(isInterSwitchFlag)
                && "Y".equals(aggregateAtPan) && intraSchemePolicy != null
                && panMaxLimit.compareTo(ZERO) > 0;

        BigDecimal panAggAmtForEntryDate = nvl(sf.getDetTradesIntraSwitchAmount())
                .add(nvl(sf.getCdtIntraSwitchAmount()))
                .add(nvl(sf.getEventBasedIntraSwitchAmount()));

        result[8] = panIntraApplies;
        result[9] = panAggAmtForEntryDate.compareTo(panMaxLimit) > 0;
        result[10] = panAggAmtForEntryDate;
        result[11] = panMaxLimit;
        result[12] = entryDate != null ? entryDate.toLocalDate().toString() : "";

        // PAN Trade/Entry Date (3830021/22)
        boolean isTradeDateAggMode = "T".equals(nvlStr(investorAggregationMode, "N"));
        boolean patWindowCheckRequired = !(patRegistrationDate != null
                && panAggSysRegnFrom != null
                && patRegistrationDate.isBefore(panAggSysRegnFrom));

        boolean panDateApplies = "Y".equals(aggregateAtPan)
                && !"M".equals(nvlStr(investorAggregationMode, "N"))
                && panMaxLimit.compareTo(ZERO) > 0
                && (isPurchase || (isSwitch && "Y".equals(isInterSwitchFlag)));

        BigDecimal panAggAmtForTradeDate = ZERO;
        if ("Y".equals(aggregateAtPan) && isTradeDateAggMode && patWindowCheckRequired) {
            if (isPurchase || (isSwitch && "Y".equals(isInterSwitchFlag))) {
                panAggAmtForTradeDate = nvl(sf.getDetTradesTradeDateAmount())
                        .add(nvl(sf.getCdtTradeDateAmount()))
                        .add(effectiveTxnAmount);
            }
        }

        BigDecimal panAggAmtForEntryDate1 = nvl(sf.getDetTradesEntryDateAmount())
                .add(nvl(sf.getCdtEntryDateAmount()))
                .add(nvl(sf.getEventBasedEntryDateAmount()));

        result[13] = panDateApplies;
        result[14] = isTradeDateAggMode;
        result[15] = patWindowCheckRequired;
        result[16] = panAggAmtForTradeDate;
        result[17] = panAggAmtForEntryDate1;
        result[18] = tradeDate != null ? tradeDate.toLocalDate().toString() : "";

        // Monthly PAN (3830023)
        boolean panMonthlyApplies = "M".equals(investorAggregationMode)
                && "Y".equals(aggregateAtPan)
                && panMaxLimit.compareTo(ZERO) > 0;

        boolean sipFlagSet = false;
        boolean stpFlagSet = false;
        if ("M".equals(investorAggregationMode)) {
            if ("AGG_SIP".equals(monthlyPanAggType)) {
                sipFlagSet = true;
            } else if ("AGG_STP".equals(monthlyPanAggType)) {
                stpFlagSet = true;
            } else if ("AGG_SIP_STP".equals(monthlyPanAggType)) {
                sipFlagSet = true;
                stpFlagSet = true;
            }
        }
        boolean skipMonthlyForTxnType = (isSwitch && sipFlagSet && !stpFlagSet)
                || (isPurchase && stpFlagSet && !sipFlagSet);

        BigDecimal panAggAmtForMonthWindow = ZERO;
        if (panMonthlyApplies) {
            BigDecimal modifiedEffectiveTxnAmount = effectiveTxnAmount;
            if (isSwitch && !"Y".equals(isInterSwitchFlag)) {
                modifiedEffectiveTxnAmount = ZERO;
            }
            panAggAmtForMonthWindow = nvl(sf.getDetTradesMonthWindowAmount())
                    .add(modifiedEffectiveTxnAmount);
        }

        String monthLabel = "";
        if (panMonthlyApplies) {
            boolean useTradeDateForMonth = !"L".equals(sfAmcCode) && !"H".equals(sfAmcCode) && !"P".equals(sfAmcCode);
            if (useTradeDateForMonth && tradeDate != null) {
                monthLabel = tradeDate.getMonth().toString().substring(0, 3) + "-" + tradeDate.getYear();
            } else if (!useTradeDateForMonth && entryDate != null) {
                monthLabel = entryDate.getMonth().toString().substring(0, 3) + "-" + entryDate.getYear();
            }
        }

        result[19] = panMonthlyApplies;
        result[20] = skipMonthlyForTxnType;
        result[21] = panAggAmtForMonthWindow;
        result[22] = monthLabel;

        return result;
    }

    public static boolean isMaxInvInvestorLimitApplies(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[0]);
    }
    public static boolean isMaxInvTotalCostExceedsLimit(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[1]);
    }
    public static boolean isMaxInvHasExceptionConfig(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[2]);
    }
    public static boolean isMaxInvWithinExceptionLimit(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[3]);
    }
    public static BigDecimal getMaxInvTotalCostValue(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[4];
    }
    public static BigDecimal getMaxInvInvestorMaxLimit(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[5];
    }
    public static BigDecimal getMaxInvCorpusPercentLimitValue(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[6];
    }
    public static BigDecimal getMaxInvExceptionTxnAmountLimit(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[7];
    }
    public static boolean isMaxInvPanIntraApplies(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[8]);
    }
    public static boolean isMaxInvPanIntraExceeded(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[9]);
    }
    public static BigDecimal getMaxInvPanAggAmtForEntryDate(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[10];
    }
    public static BigDecimal getMaxInvPanMaxLimit(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[11];
    }
    public static String getMaxInvEntryDateStr(RuleEngineRequest req) {
        return (String) computeMaxInvestmentInternal(req)[12];
    }
    public static boolean isMaxInvPanDateApplies(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[13]);
    }
    public static boolean isMaxInvTradeDateMode(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[14]);
    }
    public static boolean isMaxInvPatWindowCheckRequired(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[15]);
    }
    public static BigDecimal getMaxInvPanAggAmtForTradeDate(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[16];
    }
    public static BigDecimal getMaxInvPanAggAmtForEntryDate1(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[17];
    }
    public static String getMaxInvTradeDateStr(RuleEngineRequest req) {
        return (String) computeMaxInvestmentInternal(req)[18];
    }
    public static boolean isMaxInvPanMonthlyApplies(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[19]);
    }
    public static boolean isMaxInvSkipMonthlyForTxnType(RuleEngineRequest req) {
        return Boolean.TRUE.equals(computeMaxInvestmentInternal(req)[20]);
    }
    public static BigDecimal getMaxInvPanAggAmtForMonthWindow(RuleEngineRequest req) {
        return (BigDecimal) computeMaxInvestmentInternal(req)[21];
    }
    public static String getMaxInvMonthLabel(RuleEngineRequest req) {
        return (String) computeMaxInvestmentInternal(req)[22];
    }

    // ---- COMPUTE METHODS (return values, no context object) ----

    static Long computeAutoTrxnNo(RuleEngineRequest req) {
        if ("C".equals(req.getContext().getIpoCdtFlag()) && req.getConsolidatedDailyTradesPrimary() != null) {
            Long inputAutoTrxnNo = req.getContext().getAutoTransactionNo();
            return inputAutoTrxnNo != null ? inputAutoTrxnNo
                    : req.getConsolidatedDailyTradesPrimary().getAutoTrxnNo();
        }
        return null;
    }

    static String computeLinkSchemeCode(RuleEngineRequest req) {
        String inputLink = req.getContext().getLinkSchemeCode();
        if (inputLink == null || inputLink.trim().isEmpty()) {
            return req.getSchemeSetup() != null ? req.getSchemeSetup().getLinkSchemeCode() : null;
        }
        return inputLink;
    }

    static String computeUnitsAmountFlag(RuleEngineRequest req, BigDecimal nav) {
        String segregatedFlag = nvlStr(req.getSchemeSetup() != null
                ? req.getSchemeSetup().getSegregatedSchemeFlag() : null, "N");

        if (nav != null && nav.compareTo(BigDecimal.valueOf(-1)) != 0
                && "N".equals(segregatedFlag)
                && nav.compareTo(ZERO) != 0) {
            BigDecimal units = req.getContext().getUnits();
            if (units != null && units.compareTo(ZERO) != 0) {
                String dbCrType = getTransactionDbCrType(req);
                if ("DP".equals(dbCrType) || "DR".equals(dbCrType)) {
                    return "A";
                } else {
                    return "U";
                }
            } else {
                BigDecimal amount = getInputAmount(req);
                if (amount != null && amount.compareTo(ZERO) != 0) {
                    return "A";
                }
            }
        }
        return null;
    }

    static String computeSubTrxnType(RuleEngineRequest req) {
        String dbCr = RuleUnitUtils.deriveDbCrType(req);
        String subTrxnType = req.getContext().getSubTransactionType();
        String transactionNature = req.getContext().getTransactionNature();
        boolean isSysPlan = isSystematicPlan(req);

        if ("P".equals(dbCr) && "M".equals(subTrxnType)) {
            return "M";
        } else if (("P".equals(dbCr) && "S".equals(transactionNature))
                || "S".equals(subTrxnType)
                || "S".equals(transactionNature)
                || "U".equals(subTrxnType)
                || isSysPlan) {
            return "S";
        } else if ("A".equals(subTrxnType)) {
            return "A";
        } else {
            return "N";
        }
    }

    static String computeNewPurchaseTypeFlag(RuleEngineRequest req) {
        if (req.getDetailedTrades() != null
                && req.getDetailedTrades().getNonBonusFolioTxnExists() != null
                && req.getDetailedTrades().getNonBonusFolioTxnExists() > 0) {
            return "N";
        }
        return "Y";
    }

    static Decuple<String, String, String, Void, Void, Void, Void, Void, Void, Void> computeAggregFlags(RuleEngineRequest req) {
        SubFundSetup sfs = req.getSubFundSetup();
        String aggregTrxnDbCrForLimit = null;
        String aggregPlusTxForLimitCheck = "Y";
        String aggregDateFlagForLimit = "T";
        if (sfs != null) {
            aggregTrxnDbCrForLimit = sfs.getAggregTrxnDbCrForLimit();
            aggregPlusTxForLimitCheck = nvlStr(sfs.getAggregPlusTxForLimitCheck(), "Y");
            aggregDateFlagForLimit = nvlStr(sfs.getAggregDateFlagForLimit(), "T");
        }

        // Excel 1.6: If AGGREG_TRXN_DB_CR_FOR_LIMIT is NULL, default to all positive DB_CR types
        if (aggregTrxnDbCrForLimit == null || aggregTrxnDbCrForLimit.trim().isEmpty()) {
            // Default: all positive transaction db_cr types (P, SI, TI etc.)
            // This matches B.12 AGGREG_TRXN_DB_CR_List which is a static list of all trxn_sign='+' db_cr values
            aggregTrxnDbCrForLimit = null; // null means "all positive types" — no filtering
        } else {
            // Excel 1.6: If only one DB_CR type configured and it doesn't match current transaction, skip aggregation
            String[] dbCrValues = aggregTrxnDbCrForLimit.split(",");
            int dbCrCount = dbCrValues.length;
            if (dbCrCount == 1) {
                String currentDbCr = RuleUnitUtils.deriveDbCrType(req);
                if (currentDbCr != null && !dbCrValues[0].trim().equals(currentDbCr)) {
                    aggregPlusTxForLimitCheck = "N";
                }
            }
        }

        return Decuple.<String, String, String, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(aggregTrxnDbCrForLimit).second(aggregPlusTxForLimitCheck).third(aggregDateFlagForLimit)
                .build();
    }

    static Decuple<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void> computeSwapAndFrequency(RuleEngineRequest req, Long autoTrxnNo) {
        RuleEngineRequestContext input = req.getContext();
        boolean isSys = isSystematicByTransactionType(req) || isSystematicPlan(req);
        String swapOption = null;
        String frequency = null;
        BigDecimal autoAmount = null;

        String userType = getUserType(req);
        ConsolidatedDailyTradesPrimary cdtForSwap = req.getConsolidatedDailyTradesPrimary();
        String srcProgram = cdtForSwap != null ? cdtForSwap.getSourceProgram() : null;
        if ("EXG".equals(userType) && "TRIGGERS".equals(srcProgram)) {
            swapOption = "X";
        } else if (!"EXG".equals(userType)
                && ("CHANNEL".equals(srcProgram) || "MFU".equals(srcProgram))) {
            swapOption = "H";
        }

        if ("C".equals(input.getIpoCdtFlag()) && isSys && autoTrxnNo != null) {
            String dbCr = RuleUnitUtils.deriveDbCrType(req);
            if ( "P".equals(dbCr) || "SO".equals(dbCr) ) {
                swapOption = input.getSwapOption();
                frequency =  input.getPeriodicity() != null ? input.getPeriodicity().getCode() : null;
                autoAmount = getInputAmount(req);
            }
        } else if (isAmcBF(req) && hasInsuranceSetup(req)) {
            frequency = "LM";
            Integer insuranceId = getInsuranceId(req);
            if (isSys && insuranceId != null && insuranceId == 1003 && autoTrxnNo == null) {
                CdtTradesSip cdtForSip = req.getCdtTradesSip();
                String sipPeriodicity = cdtForSip != null ? cdtForSip.getPeriodicity() : null;
                frequency = (sipPeriodicity != null && !sipPeriodicity.trim().isEmpty()) ? sipPeriodicity : "OM";
            }
        }

        // Excel 1.7: Override for Pre-Validation with SIP parameters
        String dbCrForSwap = RuleUnitUtils.deriveDbCrType(req);
        if (isSys && ("P".equals(dbCrForSwap) || "SO".equals(dbCrForSwap) || "SI".equals(dbCrForSwap))
                && autoTrxnNo == null
                && input.getSwapOption() != null
                && frequency == null && swapOption == null && autoAmount == null) {
            frequency = input.getPeriodicity() != null ? input.getPeriodicity().getCode() : null;
            autoAmount = getInputAmount(req);
            swapOption = input.getSwapOption();
        }

        return Decuple.<String, String, BigDecimal, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(swapOption).second(frequency).third(autoAmount)
                .build();
    }

    static ZonedDateTime computeSysRegDate(RuleEngineRequest req, String subTrxnType) {
        if (!"S".equals(subTrxnType)) return null;

        // Excel 1.8: Check SYS_VALIDATE from scheme_setup
        String sysValidate = req.getSchemeSetup() != null
                ? nvlStr(req.getSchemeSetup().getSysValidate(), "T") : "T";

        if ("T".equals(sysValidate)) {
            return null;
        } else if ("R".equals(sysValidate)) {
            ZonedDateTime sysRegDate;
            if (req.getContext().getEntryDate() != null) {
                sysRegDate = req.getContext().getEntryDate();
            } else {
                sysRegDate = req.getConsolidatedDailyTrades() != null
                        ? req.getConsolidatedDailyTrades().getGdateOfBirth() : null;
            }

            // Override for pre-validation with Chan_Exg_Reg_Date
            ConsolidatedDailyTrades dailyTrades = req.getConsolidatedDailyTrades();
            if (dailyTrades != null) {
                ZonedDateTime chanExgRegDate = dailyTrades.getGdateOfBirth();
                if (chanExgRegDate != null) {
                    sysRegDate = chanExgRegDate;
                }
            }
            return sysRegDate;
        }
        return null;
    }

    static String computeApplyLimits(RuleEngineRequest req, Long autoTrxnNo, String swapOption,
                                     String sourceProgram, String userType, String dpIdRequired, String motherName) {
        SubTransactionTypesMaster stm = req.getSubTransactionTypesMaster();
        String applyLimits = stm != null ? stm.getApplyLimits() : null;

        RuleEngineRequestContext input = req.getContext();
        boolean isSys = isSystematicByTransactionType(req) || isSystematicPlan(req);
        ConsolidatedDailyTradesPrimary cdt = req.getConsolidatedDailyTradesPrimary();

        if (isSys && autoTrxnNo != null && sourceProgram != null
                && sourceProgram.startsWith("Investrak")) {
            if ("SO".equals(RuleUnitUtils.deriveDbCrType(req))) {
                if ("T".equals(swapOption) || "Q".equals(swapOption)) {
                    applyLimits = "X";
                }
            }
        }

        Integer insuranceId = getInsuranceId(req);
        if (isAmcBF(req) && insuranceId != null && insuranceId == 1003
                && ("SO".equals(RuleUnitUtils.deriveDbCrType(req)) || "SI".equals(RuleUnitUtils.deriveDbCrType(req)))) {
            applyLimits = "X";
        }

        String amcCode = getAmcCode(req);
        if ("L".equals(amcCode) && "EXG".equals(userType)
                && "Y".equals(dpIdRequired)
                && ("R".equals(RuleUnitUtils.deriveDbCrType(req)) || "SO".equals(RuleUnitUtils.deriveDbCrType(req)))) {
            applyLimits = "X";
        }

        if ("B".equals(amcCode) && isFolioExists(req)) {
            if (req.getEmployeeMaster() != null || req.getCompanyMaster() != null) {
                applyLimits = "X";
            }
        }

        if ("G".equals(amcCode)) {
            TransactionType tt = input.getTransactionType();
            if (tt != null) {
                String ttStr = tt.name();
                if ("SOCO".equals(ttStr) || "SOCO1".equals(ttStr)
                        || "SICO".equals(ttStr) || "SICO1".equals(ttStr)) {
                    applyLimits = "X";
                }
            }
        }

        if (motherName != null
                && ("BOSTP".equals(motherName) || "VLSIP".equals(motherName))) {
            applyLimits = "X";
        }

        // Excel 1.10: Smart STP package check — verify package_type = 'SMART_STP'
        if ("O".equals(amcCode) && cdt != null && cdt.getPackageName() != null) {
            if (req.getPackageInvestmentSetup() != null
                    && (req.getPackageInvestmentSetup().getSmartStpPackageCount())>0) {
                applyLimits = "X";
            }
        }

        if (cdt != null && "KEYEMPINVST".equals(cdt.getApplicationNo())) {
            if (req.getKeyEmployeeDetails() != null) {
                applyLimits = "X";
            }
        }

        if ("B".equals(amcCode) && "P".equals(RuleUnitUtils.deriveDbCrType(req))) {
            String allowLumpsum = req.getSubFundSetup() != null
                    ? req.getSubFundSetup().getAllowLumpsumOnlyForKeyEmployeesFlag() : null;
            if ("Y".equals(allowLumpsum) && req.getKeyEmployeeDetails() != null) {
                applyLimits = "X";
            }
        }

        return applyLimits;
    }

    static Decuple<String, BigDecimal, BigDecimal, ZonedDateTime, BigDecimal, BigDecimal,
            Void, Void, Void, Void> computeSwitchOutVars(RuleEngineRequest req) {
        String dbCr = RuleUnitUtils.deriveDbCrType(req);
        String soSchemeCode = null;
        BigDecimal soUnits = ZERO;
        BigDecimal soAmount = ZERO;
        ZonedDateTime soTradeDate = null;
        BigDecimal soNav = ZERO;
        BigDecimal soAmountDet = ZERO;

        if (("SI".equals(dbCr) || "TI".equals(dbCr)) && isFolioExists(req)) {
            ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
            ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
            if (cdt != null && cdt.getSoSchemeCode() != null && cdtp != null) {
                soSchemeCode = cdt.getSoSchemeCode();
                soUnits = nvl(cdtp.getUnits());
                soAmount = nvl(cdtp.getAmount());
                soTradeDate = cdt.getSwitchOutTradeDate();

                NavMaster nm = req.getNavMaster();
                if (nm != null && nm.getSoNav() != null) {
                    soNav = nm.getSoNav();
                }

                soAmountDet = nvl(cdt.getPurchaseAmountDet());
            } else if (cdt != null) {
                // DET fallback: use DET-sourced fields when CDT SO scheme code is absent
                soUnits = nvl(cdt.getSoUnitsDet());
                soAmountDet = nvl(cdt.getPurchaseAmountDet());
            }
        }

        return Decuple.<String, BigDecimal, BigDecimal, ZonedDateTime, BigDecimal, BigDecimal,
                Void, Void, Void, Void>builder()
                .first(soSchemeCode).second(soAmount).third(soUnits)
                .fourth(soTradeDate).fifth(soNav).sixth(soAmountDet)
                .build();
    }

    static Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> computeRedemptionAmounts(RuleEngineRequest req) {
        BigDecimal sumCdt;
        BigDecimal sumDet = ZERO;

        if (isFolioExists(req)) {
            ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
            ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
            if (cdt != null && cdt.getSameDayRedemptionAmount() != null) {
                sumCdt = nvl(cdt.getSameDayRedemptionAmount());
                sumDet = nvl(cdt.getRedemptionAmountDet());
            } else if ("C".equals(req.getContext().getIpoCdtFlag()) && cdtp != null) {
                sumCdt = nvl(cdtp.getAmount());
            } else {
                sumCdt = nvl(getInputAmount(req));
            }
        } else {
            sumCdt = nvl(getInputAmount(req));
        }

        return Decuple.<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(sumCdt).second(sumDet).build();
    }

    static BigDecimal computeCurrentRedemptionAmount(RuleEngineRequest req) {
        if (isFolioExists(req)) {
            ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
            if ("C".equals(req.getContext().getIpoCdtFlag()) && cdtp != null) {
                return cdtp.getAmount() != null ? nvl(cdtp.getAmount()) : nvl(getInputAmount(req));
            }
            return nvl(getInputAmount(req));
        }
        return nvl(getInputAmount(req));
    }

    static Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> computePurchaseAmounts(
            RuleEngineRequest req, BigDecimal nav, String newPurchaseTypeFlag,
            String aggregPlusTxForLimitCheck, BigDecimal soAmount, BigDecimal soAmountDet) {
        String dbCr = RuleUnitUtils.deriveDbCrType(req);
        BigDecimal sumCdt;
        BigDecimal sumDet;

        if (isFolioExists(req) && "N".equals(newPurchaseTypeFlag)) {
            if ("TI".equals(dbCr) || "SI".equals(dbCr)) {
                sumCdt = nvl(soAmount);
                sumDet = nvl(soAmountDet);
            } else {
                ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
                sumCdt = cdt != null ? nvl(cdt.getSameDayPurchaseAmount()) : ZERO;
                sumDet = cdt != null ? nvl(cdt.getPurchaseAmountDet()) : ZERO;

                if (sumCdt.compareTo(ZERO) > 0 && "S".equals(req.getContext().getIpoCdtFlag())) {
                    sumCdt = sumCdt.add(nvl(getInputAmount(req)));
                }
                if (sumCdt.compareTo(ZERO) == 0) {
                    sumCdt = nvl(getInputAmount(req));
                }
            }
        } else {
            sumCdt = nvl(getInputAmount(req));
            sumDet = ZERO;
        }

        if ("N".equals(aggregPlusTxForLimitCheck)) {
            sumCdt = nvl(getInputAmount(req));
            sumDet = ZERO;
        }

        return Decuple.<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(sumCdt).second(sumDet).build();
    }

    static BigDecimal computeCurrentPurchaseAmount(RuleEngineRequest req, BigDecimal nav,
                                                    BigDecimal soAmount, BigDecimal soUnits, BigDecimal soNav) {
        String dbCr = RuleUnitUtils.deriveDbCrType(req);
        if (isFolioExists(req)) {
            if ("SI".equals(dbCr) || "TI".equals(dbCr)) {
                BigDecimal calc = nvl(soAmount).add(nvl(soUnits).multiply(nvl(soNav)));
                int scale = getDecimalAmount(req);
                return calc.setScale(scale, RoundingMode.HALF_UP);
            } else {
                String ipoCdtFlag = req.getContext().getIpoCdtFlag();
                if ("C".equals(ipoCdtFlag)) {
                    ConsolidatedDailyTradesPrimary cdt = req.getConsolidatedDailyTradesPrimary();
                    return cdt != null && cdt.getAmount() != null
                            ? nvl(cdt.getAmount()) : nvl(getInputAmount(req));
                } else if ("I".equals(ipoCdtFlag)) {
                    int scale = getDecimalAmount(req);
                    BigDecimal calc = nvl(getInputAmount(req)).add(
                            nvl(req.getContext().getUnits()).multiply(nvl(nav)));
                    return calc.setScale(scale, RoundingMode.HALF_UP);
                }
                return nvl(getInputAmount(req));
            }
        }
        return nvl(getInputAmount(req));
    }

    static Decuple<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void> computePurchaseUnits(
            RuleEngineRequest req, BigDecimal nav, String newPurchaseTypeFlag,
            String aggregPlusTxForLimitCheck, BigDecimal soUnits, BigDecimal soNav, BigDecimal soAmount) {
        BigDecimal sumCdt = ZERO;
        BigDecimal sumDet = ZERO;
        String dbCr = RuleUnitUtils.deriveDbCrType(req);

        if (isFolioExists(req) && "N".equals(newPurchaseTypeFlag)) {
            if ("SI".equals(dbCr) || "TI".equals(dbCr)) {
                ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
                BigDecimal soUnitsDet = cdt != null ? nvl(cdt.getSoUnitsDet()) : ZERO;
                BigDecimal soAmountDet = cdt != null ? nvl(cdt.getPurchaseAmountDet()) : ZERO;
                if (soUnits != null && nvl(soUnits).compareTo(ZERO) > 0 && safeNav(nav)) {
                    sumCdt = soUnits.multiply(nvl(soNav)).divide(nav, 10, RoundingMode.HALF_UP);
                } else if (soAmount != null && soAmount.compareTo(ZERO) > 0 && safeNav(nav)) {
                    sumCdt = soAmount.divide(nav, 10, RoundingMode.HALF_UP);
                } else if (soUnitsDet.compareTo(ZERO) > 0 && safeNav(nav)) {
                    sumDet = soUnitsDet.multiply(nvl(soNav)).divide(nav, 10, RoundingMode.HALF_UP);
                } else if (soAmountDet.compareTo(ZERO) > 0 && safeNav(nav)) {
                    sumDet = soAmountDet.divide(nav, 10, RoundingMode.HALF_UP);
                }
            } else {
                ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
                sumCdt = cdt != null ? nvl(cdt.getSameDayPurchaseUnits()) : ZERO;
                sumDet = cdt != null ? nvl(cdt.getPurchaseUnitsDet()) : ZERO;
            }

            if ("S".equals(req.getContext().getIpoCdtFlag()) && safeNav(nav)) {
                BigDecimal sipUnitsAdd = nvl(getInputAmount(req)).divide(nav, 10, RoundingMode.HALF_UP);
                if (sumCdt.compareTo(ZERO) > 0) {
                    sumCdt = sumCdt.add(sipUnitsAdd);
                } else {
                    sumCdt = sipUnitsAdd;
                }
            }
        } else {
            if ("TI".equals(dbCr)) {
                sumCdt = nvl(soUnits);
            } else {
                sumCdt = nvl(getInputAmount(req));
            }
        }

        if ("N".equals(aggregPlusTxForLimitCheck)) {
            sumCdt = nvl(getInputAmount(req));
            sumDet = ZERO;
        }

        return Decuple.<BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(sumCdt).second(sumDet).build();
    }

    static BigDecimal computeCurrentPurchaseUnits(RuleEngineRequest req, BigDecimal nav,
                                                   BigDecimal soAmount, BigDecimal soUnits, BigDecimal soNav) {
        String dbCr = RuleUnitUtils.deriveDbCrType(req);
        if (isFolioExists(req)) {
            if ("SI".equals(dbCr) || "TI".equals(dbCr)) {
                if (safeNav(nav)) {
                    return nvl(soAmount).divide(nav, 10, RoundingMode.HALF_UP)
                            .add(nvl(soUnits).multiply(nvl(soNav))
                                    .divide(nav, 10, RoundingMode.HALF_UP));
                }
                return ZERO;
            } else {
                String ipoCdtFlag = req.getContext().getIpoCdtFlag();
                if ("C".equals(ipoCdtFlag)) {
                    ConsolidatedDailyTradesPrimary cdt = req.getConsolidatedDailyTradesPrimary();
                    if (cdt != null && safeNav(nav)) {
                        return nvl(cdt.getAmount()).divide(nav, 10, RoundingMode.HALF_UP)
                                .add(nvl(cdt.getUnits()));
                    }
                    return ZERO;
                } else if ("I".equals(ipoCdtFlag) || "S".equals(ipoCdtFlag)) {
                    if (safeNav(nav)) {
                        return nvl(getInputAmount(req)).divide(nav, 10, RoundingMode.HALF_UP)
                                .add(nvl(req.getContext().getUnits()));
                    }
                    return ZERO;
                }
                return ZERO;
            }
        }
        if ("TI".equals(dbCr)) {
            return nvl(soUnits);
        }
        return nvl(getInputAmount(req));
    }

    static Decuple<BigDecimal, BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void> computeRedemptionUnits(RuleEngineRequest req) {
        BigDecimal sumCdt;
        BigDecimal sumDet = ZERO;
        BigDecimal currentUnits;

        if (isFolioExists(req)) {
            ConsolidatedDailyTrades cdt = req.getConsolidatedDailyTrades();
            ConsolidatedDailyTradesPrimary cdtp = req.getConsolidatedDailyTradesPrimary();
            if (cdt != null && cdt.getSameDayRedemptionUnits() != null) {
                sumCdt = nvl(cdt.getSameDayRedemptionUnits());
                sumDet = nvl(cdt.getRedemptionUnitsDet());
            } else if ("C".equals(req.getContext().getIpoCdtFlag()) && cdtp != null) {
                sumCdt = nvl(cdtp.getUnits());
            } else {
                sumCdt = nvl(req.getContext().getUnits());
            }
            if ("C".equals(req.getContext().getIpoCdtFlag()) && cdtp != null) {
                currentUnits = nvl(cdtp.getUnits());
            } else {
                currentUnits = nvl(req.getContext().getUnits());
            }
        } else {
            sumCdt = nvl(getInputAmount(req));
            currentUnits = nvl(getInputAmount(req));
        }

        return Decuple.<BigDecimal, BigDecimal, BigDecimal, Void, Void, Void, Void, Void, Void, Void>builder()
                .first(sumCdt).second(sumDet).third(currentUnits).build();
    }

    static boolean computeInvestorPlanExists(RuleEngineRequest req) {
        InvestorPlan ip = req.getInvestorPlan();
        return ip != null && ip.getIpRecordExistsTrxn() != null && ip.getIpRecordExistsTrxn();
    }

    // ---- HIERARCHICAL LIMIT FETCHING ----
    // Returns: limitsFound, minValue, maxValue, newMinValue, newMaxValue, multiples, minUnits, maxUnits, maxPerc, schFundSubfund
    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> fetchLimitValues(RuleEngineRequest req, String subTrxnType, BigDecimal nav) {

        Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String> result;

        result = tryInsuranceLimitValues(req);
        if (result != null) return result;

        result = trySystematicLimitValues(req);
        if (result != null) return result;

        result = trySchemeLimitsHistoryValues(req, subTrxnType);
        if (result != null) return result;

        result = trySchemeLimitsCurrentValues(req, subTrxnType);
        if (result != null) return result;

        result = trySubFundLimitValues(req, subTrxnType);
        if (result != null) return result;

        result = tryFundLimitValues(req, subTrxnType);
        if (result != null) return result;

        return Decuple.<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String>builder()
                .first(false).second(ZERO).third(ZERO).fourth(ZERO).fifth(ZERO)
                .sixth(ZERO).seventh(ZERO).eighth(ZERO).ninth(ZERO).tenth("S")
                .build();
    }

    // Returns: includeCurrentBalance, aggrUnits, aggrAmount, aggrClosingBal, tooManySchemeRows, tooManySubFundRows, tooManyFundRows
    static Decuple<String, BigDecimal, BigDecimal, BigDecimal, Boolean, Boolean,
            Boolean, Void, Void, Void> fetchLimitMetadata(RuleEngineRequest req, String subTrxnType, BigDecimal nav) {
        String includeCurrentBalance = "N";
        BigDecimal aggrUnits = ZERO;
        BigDecimal aggrAmount = ZERO;
        BigDecimal aggrClosingBal = ZERO;
        boolean tooManySchemeRows = false;
        boolean tooManySubFundRows = false;
        boolean tooManyFundRows = false;

        if (isAmcBF(req) && req.getInsuranceSchemeLimits() != null) {
            return buildLimitMetadata(
                    includeCurrentBalance,
                    aggrUnits,
                    aggrAmount,
                    aggrClosingBal,
                    false,
                    false,
                    false);
        }

        boolean isSys = isSystematicByTransactionType(req) || isSystematicPlan(req);
        String periodicity = req.getContext().getPeriodicity() != null
                ? req.getContext().getPeriodicity().getCode() : null;
        if (periodicity != null && isSys && !hasInsuranceSetup(req)) {
            if (req.getSysSchemeLimitsHistory() != null || req.getSysSchemeLimits() != null) {
                return buildLimitMetadata("N", aggrUnits, aggrAmount, aggrClosingBal, false, false, false);
            }
        }

        SchemeLimitsHistory slh = req.getSchemeLimitsHistory();
        if (slh != null) {
            includeCurrentBalance = nvlStr(slh.getIncludeCurrentBalance(), "N");
            if ("Y".equals(includeCurrentBalance)) {
                aggrUnits = nvl(getFolioClosingBalance(req));
                aggrAmount = aggrUnits.multiply(nvl(nav));
                aggrClosingBal = aggrUnits;
            }
            return buildLimitMetadata(includeCurrentBalance, aggrUnits, aggrAmount, aggrClosingBal, false, false, false);
        }

        String schemeCode = req.getContext().getScheme() != null
                ? req.getContext().getScheme().getSchemeCode() : null;
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        List<SchemeLimits> schemeLevel = filterBySchemeCode(slList, schemeCode);

        if (!schemeLevel.isEmpty()) {
            if (schemeLevel.size() > 1) {
                tooManySchemeRows = true;
                return buildLimitMetadata("N", ZERO, ZERO, ZERO, true, false, false);
            }
            SchemeLimits sl = schemeLevel.get(0);
            includeCurrentBalance = nvlStr(sl.getIncludeCurrentBalance(), "N");
            if ("Y".equals(includeCurrentBalance)) {
                aggrUnits = nvl(getFolioClosingBalance(req));
                aggrAmount = aggrUnits.multiply(nvl(nav));
                aggrClosingBal = aggrUnits;
            }
            return buildLimitMetadata(includeCurrentBalance, aggrUnits, aggrAmount, aggrClosingBal, false, false, false);
        }

        SubFundSetup sfs = req.getSubFundSetup();
        if (sfs != null && "Y".equals(sfs.getAggregateFlag())) {
            SchemeControls sc = req.getSchemeControls();
            if (sc != null) {
                aggrUnits = nvl(sc.getSubFundClosingBalance());
                aggrAmount = aggrUnits.multiply(nvl(nav));
                aggrClosingBal = aggrUnits;
            }
            String subFundCode = req.getSchemeSetup() != null
                    ? req.getSchemeSetup().getParentSubFundCode() : null;
            if (subFundCode != null && !subFundCode.isEmpty() && slList != null) {
                List<SchemeLimits> subFundLevel = filterBySchemeCode(slList, subFundCode);
                if (subFundLevel.size() > 1) {
                    return buildLimitMetadata("N", aggrUnits, aggrAmount, aggrClosingBal, false, true, false);
                }
            }
            return buildLimitMetadata("N", aggrUnits, aggrAmount, aggrClosingBal, false, false, false);
        }

        SchemeSetup ss = req.getSchemeSetup();
        if (ss != null && ss.getParentFundCode() != null) {
            String fundAggregateFlag = req.getSchemeMaster() != null
                    ? req.getSchemeMaster().getAggregateFlag() : "N";
            if ("Y".equals(fundAggregateFlag)) {
                SchemeControls sc = req.getSchemeControls();
                if (sc != null) {
                    aggrUnits = nvl(sc.getFundClosingBalance());
                    aggrAmount = aggrUnits.multiply(nvl(nav));
                    aggrClosingBal = aggrUnits;
                }
                String fundCode = ss.getParentFundCode();
                if (slList != null) {
                    List<SchemeLimits> fundLevel = filterBySchemeCode(slList, fundCode);
                    if (fundLevel.size() > 1) {
                        return buildLimitMetadata("N", aggrUnits, aggrAmount, aggrClosingBal, false, false, true);
                    }
                }
                return buildLimitMetadata("N", aggrUnits, aggrAmount, aggrClosingBal, false, false, false);
            }
        }

        return buildLimitMetadata("N", ZERO, ZERO, ZERO, false, false, false);
    }

    private static Decuple<String, BigDecimal, BigDecimal, BigDecimal, Boolean, Boolean,
            Boolean, Void, Void, Void> buildLimitMetadata(String includeCurrentBalance,
            BigDecimal aggrUnits, BigDecimal aggrAmount, BigDecimal aggrClosingBal,
            boolean tooManyScheme, boolean tooManySubFund, boolean tooManyFund) {
        return Decuple.<String, BigDecimal, BigDecimal, BigDecimal, Boolean, Boolean,
                Boolean, Void, Void, Void>builder()
                .first(includeCurrentBalance).second(aggrUnits).third(aggrAmount).fourth(aggrClosingBal)
                .fifth(tooManyScheme).sixth(tooManySubFund).seventh(tooManyFund)
                .build();
    }

    // ---- Try methods for hierarchical limit values ----

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> tryInsuranceLimitValues(RuleEngineRequest req) {
        if (!isAmcBF(req) || req.getInsuranceSchemeLimits() == null) {
            return null;
        }
        InsuranceSchemeLimits isl = req.getInsuranceSchemeLimits();
        if (isl == null) return null;

        return buildLimitValues(true, nvl(isl.getMinAmount()), nvl(isl.getMaxAmount()),
                nvl(isl.getMinAmount()), nvl(isl.getMaxAmount()), nvl(isl.getMultiples()),
                nvl(isl.getMinUnits()), nvl(isl.getMaxUnits()), ZERO, "S");
    }

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> trySystematicLimitValues(RuleEngineRequest req) {
        boolean isSys = isSystematicByTransactionType(req) || isSystematicPlan(req);
        String periodicity = req.getContext().getPeriodicity() != null
                ? req.getContext().getPeriodicity().getCode() : null;

        if (periodicity == null || !isSys) return null;
        if (hasInsuranceSetup(req)) return null;

        SysSchemeLimitsHistory sslh = req.getSysSchemeLimitsHistory();
        if (sslh != null) {
            return buildLimitValues(true, nvl(sslh.getMinAmount()), nvl(sslh.getMaxAmount()),
                    nvl(sslh.getMinAmount()), nvl(sslh.getMaxAmount()), nvl(sslh.getMultiples()),
                    nvl(sslh.getMinUnits()), nvl(sslh.getMaxUnits()), ZERO, "S");
        }

        SysSchemeLimits ssl = req.getSysSchemeLimits();
        if (ssl != null) {
            return buildLimitValues(true, nvl(ssl.getMinAmount()), nvl(ssl.getMaxAmount()),
                    nvl(ssl.getMinAmount()), nvl(ssl.getMaxAmount()), nvl(ssl.getMultiples()),
                    nvl(ssl.getMinUnits()), nvl(ssl.getMaxUnits()), ZERO, "S");
        }
        return null;
    }

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> trySchemeLimitsHistoryValues(RuleEngineRequest req, String subTrxnType) {
        SchemeLimitsHistory slh = req.getSchemeLimitsHistory();
        if (slh == null) return null;

        BigDecimal multiples = "S".equals(subTrxnType) ? nvl(slh.getSipMultiples()) : nvl(slh.getMultiples());
        BigDecimal minValue = nvl(slh.getMinValue());
        BigDecimal maxValue = nvl(slh.getMaxValue());
        BigDecimal newMinValue = nvl(slh.getNewPurchMinvalue());
        BigDecimal newMaxValue = nvl(slh.getNewPurchMaxvalue());
        BigDecimal minUnits = nvl(slh.getMinUnits());
        BigDecimal maxUnits = nvl(slh.getMaxUnits());
        BigDecimal maxPerc = nvl(slh.getMaxPercentage());

        BigDecimal minSwp = nvl(slh.getMinSwpAmount());
        BigDecimal maxSwp = nvl(slh.getMaxSwpAmount());
        BigDecimal microMin = nvl(slh.getMicroSipMinimum());
        BigDecimal microMax = nvl(slh.getMicroSipMaximum());
        BigDecimal minSysU = nvl(slh.getMinSysUnits());
        BigDecimal maxSysU = nvl(slh.getMaxSysUnits());

        Decuple<BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                Void, Void, Void, Void> adjusted = applySystematicOverrides(subTrxnType, minValue, maxValue,
                newMinValue, newMaxValue, minUnits, maxUnits, minSwp, maxSwp, microMin, microMax, minSysU, maxSysU);

        return buildLimitValues(true, adjusted.getFirst(), adjusted.getSecond(),
                adjusted.getThird(), adjusted.getFourth(), multiples,
                adjusted.getFifth(), adjusted.getSixth(), maxPerc, "S");
    }

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> trySchemeLimitsCurrentValues(RuleEngineRequest req, String subTrxnType) {
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        if (slList == null || slList.isEmpty()) return null;

        String schemeCode = req.getContext().getScheme() != null
                ? req.getContext().getScheme().getSchemeCode() : null;
        List<SchemeLimits> schemeLevel = filterBySchemeCode(slList, schemeCode);

        if (schemeLevel.isEmpty()) return null;
        if (schemeLevel.size() > 1) {
            return buildLimitValues(true, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, "S");
        }

        SchemeLimits sl = schemeLevel.get(0);
        return extractSchemeLimitsValues(sl, subTrxnType, "S");
    }

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> trySubFundLimitValues(RuleEngineRequest req, String subTrxnType) {
        SubFundSetup sfs = req.getSubFundSetup();
        if (sfs == null || !"Y".equals(sfs.getAggregateFlag())) return null;

        String subFundCode = req.getSchemeSetup() != null
                ? req.getSchemeSetup().getParentSubFundCode() : null;
        if (subFundCode == null || subFundCode.isEmpty()) return null;

        List<SchemeLimits> slList = req.getSchemeLimitsList();
        if (slList != null && !slList.isEmpty()) {
            List<SchemeLimits> subFundLevel = filterBySchemeCode(slList, subFundCode);

            if (subFundLevel.size() == 1) {
                return extractSchemeLimitsValues(subFundLevel.get(0), subTrxnType, "B");
            }
            if (subFundLevel.size() > 1) {
                return buildLimitValues(true, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, "B");
            }
        }
        return null;
    }

    static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> tryFundLimitValues(RuleEngineRequest req, String subTrxnType) {
        SchemeSetup ss = req.getSchemeSetup();
        if (ss == null || ss.getParentFundCode() == null) return null;

        String fundAggregateFlag = req.getSchemeMaster() != null
                ? req.getSchemeMaster().getAggregateFlag() : "N";
        if (!"Y".equals(fundAggregateFlag)) return null;

        String fundCode = ss.getParentFundCode();
        List<SchemeLimits> slList = req.getSchemeLimitsList();
        if (slList != null && !slList.isEmpty()) {
            List<SchemeLimits> fundLevel = filterBySchemeCode(slList, fundCode);

            if (fundLevel.size() == 1) {
                return extractSchemeLimitsValues(fundLevel.get(0), subTrxnType, "F");
            }
            if (fundLevel.size() > 1) {
                return buildLimitValues(true, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, "F");
            }
        }
        return null;
    }

    private static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> extractSchemeLimitsValues(SchemeLimits sl, String subTrxnType, String schFundSubfund) {
        BigDecimal multiples = "S".equals(subTrxnType) ? nvl(sl.getSipMultiples()) : nvl(sl.getMultiples());
        BigDecimal minValue = nvl(sl.getMinValue());
        BigDecimal maxValue = nvl(sl.getMaxValue());
        BigDecimal newMinValue = nvl(sl.getNewPurchMinvalue());
        BigDecimal newMaxValue = nvl(sl.getNewPurchMaxvalue());
        BigDecimal minUnits = nvl(sl.getMinUnits());
        BigDecimal maxUnits = nvl(sl.getMaxUnits());
        BigDecimal maxPerc = nvl(sl.getMaxPercentage());

        BigDecimal minSwp = nvl(sl.getMinSwpAmount());
        BigDecimal maxSwp = nvl(sl.getMaxSwpAmount());
        BigDecimal microMin = nvl(sl.getMicroSipMinimum());
        BigDecimal microMax = nvl(sl.getMicroSipMaximum());
        BigDecimal minSysU = nvl(sl.getMinSysUnits());
        BigDecimal maxSysU = nvl(sl.getMaxSysUnits());

        Decuple<BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                Void, Void, Void, Void> adjusted = applySystematicOverrides(subTrxnType, minValue, maxValue,
                newMinValue, newMaxValue, minUnits, maxUnits, minSwp, maxSwp, microMin, microMax, minSysU, maxSysU);

        return buildLimitValues(true, adjusted.getFirst(), adjusted.getSecond(),
                adjusted.getThird(), adjusted.getFourth(), multiples,
                adjusted.getFifth(), adjusted.getSixth(), maxPerc, schFundSubfund);
    }

    // Returns: minValue, maxValue, newMinValue, newMaxValue, minUnits, maxUnits
    private static Decuple<BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            Void, Void, Void, Void> applySystematicOverrides(String subTrxnType,
            BigDecimal minValue, BigDecimal maxValue, BigDecimal newMinValue, BigDecimal newMaxValue,
            BigDecimal minUnits, BigDecimal maxUnits,
            BigDecimal minSwp, BigDecimal maxSwp, BigDecimal microMin, BigDecimal microMax,
            BigDecimal minSysU, BigDecimal maxSysU) {
        if ("M".equals(subTrxnType)) {
            minValue = microMin;
            maxValue = microMax;
            newMinValue = minValue;
            newMaxValue = maxValue;
        } else if ("S".equals(subTrxnType) || "A".equals(subTrxnType)) {
            minValue = minSwp;
            maxValue = maxSwp;
            newMinValue = minValue;
            newMaxValue = maxValue;
            if ("S".equals(subTrxnType)) {
                minUnits = minSysU;
                maxUnits = maxSysU;
            }
        }

        return Decuple.<BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                Void, Void, Void, Void>builder()
                .first(minValue).second(maxValue).third(newMinValue).fourth(newMaxValue)
                .fifth(minUnits).sixth(maxUnits).build();
    }

    private static Decuple<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
            BigDecimal, BigDecimal, BigDecimal, String> buildLimitValues(boolean found,
            BigDecimal minValue, BigDecimal maxValue, BigDecimal newMinValue, BigDecimal newMaxValue,
            BigDecimal multiples, BigDecimal minUnits, BigDecimal maxUnits, BigDecimal maxPerc,
            String schFundSubfund) {
        return Decuple.<Boolean, BigDecimal, BigDecimal, BigDecimal, BigDecimal, BigDecimal,
                BigDecimal, BigDecimal, BigDecimal, String>builder()
                .first(found).second(minValue).third(maxValue).fourth(newMinValue).fifth(newMaxValue)
                .sixth(multiples).seventh(minUnits).eighth(maxUnits).ninth(maxPerc).tenth(schFundSubfund)
                .build();
    }

    static List<SchemeLimits> filterBySchemeCode(List<SchemeLimits> list, String code) {
        if (code == null || list == null) return List.of();
        return list.stream()
                .filter(sl -> code.equals(sl.getSchemeCode()))
                .toList();
    }

    private static int getSifFolioExistsCount(RuleEngineRequest req) {
        String otherAmc = getApplicationParameter(req, "OTHER_AMC");
        if (!"SIF".equals(otherAmc)) return 0;
        if (req.getContext().getMinorFlag() == null) return 0;
        InvestorPlan ip = req.getInvestorPlan();
        if (ip == null) return 0;
        if (ip.getClosingBalance() != null && ip.getClosingBalance().compareTo(ZERO) > 0) {
            return 1;
        }
        return 0;
    }

    // ---- HELPER METHODS ----

    public static BigDecimal nvl(BigDecimal val) {
        return val != null ? val : ZERO;
    }

    private static String nvlStr(String val, String def) {
        return val != null && !val.trim().isEmpty() ? val : def;
    }

    private static boolean safeNav(BigDecimal nav) {
        return nav != null && nav.compareTo(ZERO) != 0;
    }

    private static BigDecimal getInputAmount(RuleEngineRequest req) {
        if (req.getContext().getAmount() != null) {
            return req.getContext().getAmount().getAmountValue();
        }
        return ZERO;
    }

    private static BigDecimal getNav(RuleEngineRequest req) {
        NavMaster nm = req.getNavMaster();
        if (nm != null && nm.getNav() != null) {
            return nm.getNav();
        }
        return ZERO;
    }

    public static String getAmcCode(RuleEngineRequest req) {
        if (req.getAmc() != null) return req.getAmc().getAmcCode();
        return null;
    }

    private static boolean isAmcBF(RuleEngineRequest req) {
        String amc = getAmcCode(req);
        return "B".equals(amc) || "F".equals(amc);
    }

    private static boolean hasInsuranceSetup(RuleEngineRequest req) {
        return req.getInsuranceSetup() != null;
    }

    private static Integer getInsuranceId(RuleEngineRequest req) {
        if (req.getInsuranceSetup() != null) {
            return req.getInsuranceSetup().getInsuranceId();
        }
        return null;
    }

    private static boolean isFolioExists(RuleEngineRequest req) {
        return req.getFolioNumberComputation() != null;
    }

    public static String getTransactionSign(RuleEngineRequest req) {
        if (req.getTransactionTypes() != null) {
            return req.getTransactionTypes().getTransactionSign();
        }
        return null;
    }

    private static String getTransactionDbCrType(RuleEngineRequest req) {
        if (req.getTransactionTypes() != null && req.getTransactionTypes().getTrxnDbCr() != null) {
            return req.getTransactionTypes().getTrxnDbCr();
        }
        return RuleUnitUtils.deriveDbCrType(req);
    }

    private static boolean isSystematicByTransactionType(RuleEngineRequest req) {
        TransactionType tt = req.getContext().getTransactionType();
        return tt == TransactionType.STOP || tt == TransactionType.STIP
                || tt == TransactionType.SWP || tt == TransactionType.SIP;
    }

    private static boolean isSystematicPlan(RuleEngineRequest req) {
        SubTransactionTypesMaster stm = req.getSubTransactionTypesMaster();
        return stm != null && stm.getSystematicFlag() != null && stm.getSystematicFlag();
    }

    private static String getUserType(RuleEngineRequest req) {
        return req.getUserMaster() != null ? nvlStr(req.getUserMaster().getUserType(), "XYZ") : "XYZ";
    }

    private static String getDpIdRequired(RuleEngineRequest req) {
        return req.getUserMaster() != null ? nvlStr(req.getUserMaster().getDpIdRequired(), "N") : "N";
    }

    private static String getCheckLimitPlus(RuleEngineRequest req) {
        PaymentMechanism pm = req.getPaymentMechanism();
        return pm != null ? nvlStr(pm.getCheckLimit(), "Y") : "Y";
    }

    private static String getCheckLimitMinus(RuleEngineRequest req) {
        PayoutMechanism pm = req.getPayoutMechanism();
        return pm != null ? nvlStr(pm.getCheckLimit(), "Y") : "Y";
    }

    public static int getDecimalAmount(RuleEngineRequest req) {
        if (req.getSchemeSetup() != null && req.getSchemeSetup().getDecimalAmount() != null) {
            return req.getSchemeSetup().getDecimalAmount();
        }
        return 2;
    }

    private static BigDecimal getSchemeControlsClosingBalance(RuleEngineRequest req) {
        SchemeControls sc = req.getSchemeControls();
        return sc != null ? nvl(sc.getClosingBalance()) : ZERO;
    }

    private static BigDecimal getFolioClosingBalance(RuleEngineRequest req) {
        InvestorPlan ip = req.getInvestorPlan();
        return ip != null ? nvl(ip.getClosingBalance()) : ZERO;
    }

    private static String getApplicationParameter(RuleEngineRequest req, String keyName) {
        if (req.getApplicationParameters() == null) return null;
        for (AmcApplicationParameter p : req.getApplicationParameters()) {
            if (p != null && keyName.equals(p.getKeyName())) {
                return p.getKeyValue();
            }
        }
        return null;
    }

    public static boolean isElssExemptFromMultiples(RuleEngineRequest req) {
        if (!"SI".equals(RuleUnitUtils.deriveDbCrType(req))) return false;

        String schemeCategory = req.getSchemeSetup() != null ? req.getSchemeSetup().getSchemeCategory() : null;
        if (!"OE".equals(schemeCategory) && !"CE".equals(schemeCategory)) return false;

        String convertElss = req.getAmc() != null ? req.getAmc().getConvertSoToElssMultiples() : null;
        String siRounded = req.getSchemeOption() != null ? req.getSchemeOption().getSiToBeRounded() : null;

        return "Y".equals(convertElss) || "Y".equals(siRounded);
    }


    private static String getInterSwitchFlag(RuleEngineRequest req) {
        // Inter-scheme switch: 'Y' if source and target schemes belong to different sub-funds, 'N' otherwise
        SubFundSetup sf = req.getSubFundSetup();
        SchemeMaster sm = req.getSchemeMaster();
        if (sf != null && sm != null && req.getContext().getToSchemeCode() != null) {
            String sourceSubFund = sm.getParentSubFundCode();
            String targetSubFund = sf.getSubFundCode();
            if (sourceSubFund != null && !sourceSubFund.equals(targetSubFund)) {
                return "Y";
            }
        }
        return "N";
    }

    private static BigDecimal getNavForCalc(RuleEngineRequest req, String schemeCode, ZonedDateTime navDate) {
        BigDecimal navVal = getNav(req);
        if (navVal.compareTo(ZERO) <= 0) return ZERO;
        return navVal;
    }
}
