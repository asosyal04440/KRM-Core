from krm.training.curriculum import assign_curriculum_stage, build_curriculum_mix, curriculum_statistics
from krm.training.record_format import TrainingTaskType, make_record


def test_curriculum_stages_and_profiles() -> None:
    records = [
        make_record(TrainingTaskType.INTENT_ROUTING, "q", "intent"),
        make_record(TrainingTaskType.ANSWER_PLANNING, "graph", "plan"),
        make_record(TrainingTaskType.GROUNDED_ANSWER, "plan", "answer"),
    ]
    assert assign_curriculum_stage(records[0]) == "CONCEPTS"
    planner = build_curriculum_mix(records, "planner_only")
    assert [record.task for record in planner] == [TrainingTaskType.ANSWER_PLANNING]
    stats = curriculum_statistics(records)
    assert stats["stages"]["CONCEPTS"] == 1
