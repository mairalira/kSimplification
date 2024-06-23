from ProtoTypes.ProtoTypes import get_prototypes

from selectBestApproximation.critieria import get_best_approximation

from utils.data import get_time_series, get_min_max_from_dataset_name
from utils.folder import make_folder
from utils.model import class_to_color

from models.loadModel import model_classify

from visualization.plotting import TSParam, PlotParams
from visualization.getTSParam import get_ts_param_org, get_ts_param_approx
from visualization.multiplePlot import MultiPlotParams

from Perturbations.dataTypes import SegmentedTS


def all_datasets_multiplot_prototypes():
    all_datasets = ["Chinatown", "ECG200"]
    all_models = ["Chinatown_100.keras", "ECG200_100.keras"]
    for dataset, model in zip(all_datasets, all_models):
        multi_plot_all_prototypes(dataset_name=dataset, model_name=model)


def multi_plot_all_prototypes(dataset_name, model_name):
    min_y, max_y = get_min_max_from_dataset_name(dataset_name=dataset_name)
    class_0, class_1 = get_prototypes(dataset_name=dataset_name, model_name=model_name)
    # class_0 = class_0[:2]
    # class_1 = class_1[:2]

    all_plot_params_clean = []
    all_plot_params_explanation = []
    for idx in class_0 + class_1:
        org_ts = get_time_series(
            dataset_name=dataset_name,
            instance_nr=idx,

        )
        class_org = model_classify(model_name=model_name, time_series=org_ts)
        ts_param_org = TSParam(
            x_values=list(range(len(org_ts))),
            y_values=org_ts,
            color=class_to_color(class_org)
        )

        ts_param_org_black = get_ts_param_org(
            y_org=org_ts,
            model_name=model_name,
            linestyle=(0, (2, 0.5)),
            linewidth=1

        )

        best_approximation = get_best_approximation(dataset_name=dataset_name, model_name=model_name, instance_nr=idx,
                                                    gamma=0.1)
        ts_param_approx = get_ts_param_approx(
            y_approx=best_approximation.line_version,
            model_name=model_name,
            linewidth=3,
            linestyle="-"
        )

        # Clean
        plot_param_clean = PlotParams(
            ts_params=[ts_param_org],
            y_lim=(min_y, max_y)
        )
        all_plot_params_clean.append(plot_param_clean)

        # Explanation
        plot_param_explanation = PlotParams(
            ts_params=[ts_param_approx, ts_param_org_black],
            y_lim=(min_y, max_y)
        )
        all_plot_params_explanation.append(plot_param_explanation)

    # Store images in folder
    location = f"PyPlots/{dataset_name}/"
    folder_name_clean = "CleanPrototypes"
    folder_name_w_simp = "ExplainedPrototypes"
    make_folder(location + folder_name_clean)
    make_folder(location + folder_name_w_simp)

    # clean
    MultiPlotParams(
        plotParams=all_plot_params_clean,
        save_file=f"{dataset_name}/{folder_name_clean}/multiPlot",
        display=True,
        rows=2,
        cols=3
    ).plot()

    # Explanation
    MultiPlotParams(
        plotParams=all_plot_params_explanation,
        save_file=f"{dataset_name}/{folder_name_w_simp}/multiPlot",
        display=True,
        rows=2,
        cols=3
    ).plot()


def plot_all_proto_types_clean():
    dataset_name = "Chinatown"
    model_name = "Chinatown_100.keras"
    min_y, max_y = get_min_max_from_dataset_name(dataset_name=dataset_name)
    class_0, class_1 = get_prototypes(dataset_name=dataset_name, model_name=model_name)
    print(class_1)
    all_approximations = []
    all_org_ts = []
    for idx in class_0 + class_1:
        best_approximation = get_best_approximation(dataset_name=dataset_name, model_name=model_name, instance_nr=idx)
        org_ts = get_time_series(dataset_name=dataset_name, instance_nr=idx)
        all_approximations.append(best_approximation)
        all_org_ts.append(org_ts)

    # Store images in folder
    location = f"PyPlots/{dataset_name}/"
    folder_name_clean = "CleanPrototypes"
    folder_name_w_simp = "ExplainedPrototypes"
    make_folder(location + folder_name_clean)
    make_folder(location + folder_name_w_simp)
    all_class_0 = []
    all_class_1 = []
    for org, idx in zip(all_org_ts, (class_0 + class_1)):
        class_current = model_classify(model_name=model_name, time_series=org)

        ts_param_org = TSParam(
            x_values=list(range(len(all_org_ts[0]))),
            y_values=org,
            color=class_to_color(
                class_current
            )
        )
        PlotParams(
            ts_params=[ts_param_org],
            save_file=f"{dataset_name}/" + folder_name_clean + f"/{idx}",
            y_lim=(min_y, max_y)
        ).make_plot()
        if class_current == 0:
            all_class_0.append(ts_param_org)
        else:
            all_class_1.append(ts_param_org)

    PlotParams(
        ts_params=all_class_0,
        save_file=f"{dataset_name}/" + folder_name_clean + f"/class_0",
        y_lim=(min_y, max_y)
    ).make_plot()
    PlotParams(
        ts_params=all_class_1,
        save_file=f"{dataset_name}/" + folder_name_clean + f"/class_1",
        y_lim=(min_y, max_y)
    ).make_plot()

    for org, approx, idx in zip(all_org_ts, all_approximations,
                                (class_0 + class_1)):  # type: List[float],SegmentedTS, int
        class_current = model_classify(model_name=model_name, time_series=org)
        ts_param_org = TSParam(
            x_values=list(range(len(all_org_ts[0]))),
            y_values=org,
            color=class_to_color(
                class_current
            )
        )
        ts_param_approx = get_ts_param_approx(
            y_approx=approx.line_version,
            model_name=model_name
        )
        PlotParams(
            ts_params=[ts_param_org, ts_param_approx],
            save_file=f"{dataset_name}/" + folder_name_w_simp + f"/{idx}",
            y_lim=(min_y, max_y)
        ).make_plot()
        if class_current == 0:
            all_class_0.append(ts_param_approx)
        else:
            all_class_1.append(ts_param_approx)

    PlotParams(
        ts_params=all_class_0,
        save_file=f"{dataset_name}/" + folder_name_w_simp + f"/class_0",
        y_lim=(min_y, max_y)
    ).make_plot()
    PlotParams(
        ts_params=all_class_1,
        save_file=f"{dataset_name}/" + folder_name_w_simp + f"/class_1",
        y_lim=(min_y, max_y)
    ).make_plot()


if __name__ == '__main__':
    # plot_all_proto_types_clean()
    all_datasets_multiplot_prototypes()
