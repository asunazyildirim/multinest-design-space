import pythoncom
import win32com.client


def get_open_hysys_case(case_path):

    pythoncom.CoInitialize()

    rot = pythoncom.GetRunningObjectTable()
    enum = rot.EnumRunning()
    bind_ctx = pythoncom.CreateBindCtx(0)

    while True:

        items = enum.Next(1)

        if not items:
            break

        moniker = items[0]

        try:

            name = moniker.GetDisplayName(
                bind_ctx,
                None
            )

            if name.lower() == case_path.lower():

                print(
                    f"Found ROT entry: {name}"
                )

                obj = rot.GetObject(moniker)

                dispatch = obj.QueryInterface(
                    pythoncom.IID_IDispatch
                )

                case = win32com.client.Dispatch(
                    dispatch
                )

                print(
                    "Connected to HYSYS case"
                )

                return case

        except Exception:
            pass


    raise RuntimeError(
        f"HYSYS case not found: {case_path}"
    )