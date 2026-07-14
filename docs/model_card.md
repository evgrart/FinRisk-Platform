# FinRisk Platform — model card

## Назначение

Модели демонстрируют инженерный контур риск-скоринга: от point-in-time
признаков до calibrated probability и решения по бизнес-порогам.

## Данные

Используются только синтетические данные, созданные `finrisk.data.generate`.
Они не описывают реальное поведение клиентов и не могут использоваться для
реальных кредитных или платёжных решений.

## Модели

- Credit risk: Logistic Regression и Random Forest; победитель выбирается по
  validation PR-AUC, затем вероятность калибруется sigmoid-калибровкой.
- Fraud: Logistic Regression и Random Forest; threshold выбирается на
  validation по стоимости ошибок, где пропуск fraud дороже false positive.

## Валидация

Данные делятся по времени на train/validation/test. Test используется только
для финальной оценки. В `artifacts/*/metrics.json` сохраняются PR-AUC, ROC-AUC,
Brier score, recall, error rates и выбранный threshold.

## Ограничения

- synthetic data не заменяет банковскую выборку, data governance и legal review;
- reason codes — прозрачные сигналы для ручной проверки, а не причинное
  объяснение модели;
- пороги являются демонстрационными и должны пересчитываться при изменении
  стоимости ошибок, продукта или risk appetite;
- перед production нужны backtesting, fairness-аудит, stress tests, drift
  monitoring и human-in-the-loop процесс.
