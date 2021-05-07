


def mu(x):
    x1, x2, x3, x4 = x

    d21 = x2 - x1
    d43 = x4 - x3
    d13 = x1 - x3

    d1321 = (d13 * d21).sum(axis=-1)
    d1343 = (d13 * d43).sum(axis=-1)
    d2121 = (d21 * d21).sum(axis=-1)
    d4321 = (d43 * d21).sum(axis=-1)
    d4343 = (d43 * d43).sum(axis=-1)
    nom = (d1343*d4321 - d1321*d4343)
    den =  (d2121*d4343 - d4321*d4321)
    mua =  nom / den
    mua = np.clip(mua, a_min=0, a_max=1)
    mub = (d1343 + mua*d4321) / d4343
    mub = np.clip(mub, a_min=0, a_max=1)
    return np.array([mua, mub])


def d2(x):
    mua, mub = mu(x)
    x1, x2, x3, x4 = x
    d21 = x2 - x1
    d43 = x4 - x3

    xa = x1 + mua*d21
    xb = x3 + mub*d43

    d = xb-xa
    return d.dot(d)


def mu_jac(x):

    x1, x2, x3, x4 = x

    d21 = x2 - x1
    d43 = x4 - x3
    d13 = x1 - x3

    d1321 = (d13 * d21).sum(axis=-1)
    d1343 = (d13 * d43).sum(axis=-1)
    d2121 = (d21 * d21).sum(axis=-1)
    d4321 = (d43 * d21).sum(axis=-1)
    d4343 = (d43 * d43).sum(axis=-1)

    a = d1343*d4321
    b = d1321*d4343
    c = d2121*d4343
    d = d4321*d4321

    f = d4321*d4343
    g = d1343/d4343
    nom = a - b
    den = c - d
    mua = nom / den
    mua = np.clip(mua, 0, 1)
    mub = (d1343 + mua*d4321) / d4343
    mub = np.clip(mub, 0, 1)

    d_mua = np.zeros_like(x)
    d_mub = np.zeros_like(x)

    if mua != 0 and mua != 1:
        temp = +d4321 * d43 - d4343 * d21
        d_mua[1] = -d4343*d13 + (d1343 + 2*mua*d4321)*d43 - 2*mua*d4343*d21
        d_mua[0] = -d_mua[1] + temp
        d_mua[3] = +d4321*d13 - (d1321 + mua*d2121)*2*d43 + (d1343 + 2*mua*d4321)*d21
        d_mua[2] = -d_mua[3] - temp
        d_mua /= den

    if mub != 0 and mub != 1:
        if mua == 0:
            d_mub[0] = d43
            d_mub[1] = 0
            d_mub[3] = d13 - 2 * g * d43
            d_mub[2] = -d_mub[3] - d_mub[0]
            d_mub /= d4343

        elif mua == 1:
            d_mub[0] = 0
            d_mub[1] = d43
            d_mub[3] = d13 + d21 - (2 * g + 2 * d4321 / d4343) * d43
            d_mub[2] = -d_mub[3] - d_mub[1]
            d_mub /= d4343

        else:
            temp = +den*d43 + d*d43 - f*d21
            # temp2 = d1343/d4343 * d43
            d_mub[1] = -f*d13 - 2*mua*f*d21 + (a + nom + 2*mua*d)*d43
            d_mub[0] = -d_mub[1] + temp
            d_mub[3] = (d + den)*d13 + (a + nom + 2*mua*d)*d21 - ((mua*d2121 + nom/d4343  + d1321)*d4321 + den*g)*2*d43
            d_mub[2] = -d_mub[3] - temp
            d_mub /= (den*d4343)


    return mua, mub, d_mua, d_mub


def d2_jac(x):
    # d = |xb - xa| = |dba|
    # d2 = d**2 = dba_x^2 + dba_y^2 + dba_z^2
    # dd2_dxb = 2*dba
    # dd2_dxa = -2*dba
    # dd2_dba = 2*dba

    # xa = x1 + mua*(x2-x1) = x1 - mua*x1 + mua*x2
    # xb = x3 + mub*(x4-x3) = x3 - mub*x3 + mub*x4

    # dxa_dx1 = 1 - mua - dmua_dx1*x1 + dmua_dx1*x2
    # dxa_dx2 = -dmua_dx2*x1 + mua + dmua_dx2*x2
    # dxa_dx3 = dmua_dx3*(x2-x1)
    # dxa_dx4 = dmua_dx4*(x2-x1)

    # dxb_dx1 = dmub_dx1*(x4-x3)
    # dxb_dx2 = dmub_dx2*(x4-x3)
    # dxb_dx3 = 1 - mub - dmub_dx3*x3 + dmub_dx3*x4
    # dxb_dx4 = -dmub_dx4*x3 + mub + dmub_dx4*x4

    # ddba_dx1 = +dmub_dx1*(x4-x3) + dmua_dx1*x1 - dmua_dx1*x2 + mua - 1
    # ddba_dx3 = -dmua_dx3*(x2-x1) - dmub_dx3*x3 + dmub_dx3*x4 - mub + 1
    # ddba_dx2 = +dmub_dx2*(x4-x3) + dmua_dx2*x1 - dmua_dx2*x2 - mua
    # ddba_dx4 = -dmua_dx4*(x2-x1) - dmub_dx4*x3 + dmub_dx4*x4 + mub

    # -dmua_dx1*d21 + dmub_dx1*d43 - (1 - mua)
    # -dmua_dx2*d21 + dmub_dx2*d43 - mua
    # -dmua_dx3*d21 + dmub_dx3*d43 + (1 - mub)
    # -dmua_dx4*d21 + dmub_dx4*d43 + mub


    x1, x2, x3, x4 = x
    d21 = x2 - x1
    d43 = x4 - x3
    mua, mub, dmua_dx, dmub_dx = mu_jac(x)
    dx_dx = np.array([[-1, 0, 1, 0]]).T
    d21_dx = np.array([[-1, 1, 0, 0]]).T
    d43_dx = np.array([[0, 0, -1, 1]]).T
    dxaxb_dx = dx_dx + dmub_dx*d43 + mub*d43_dx - (dmua_dx*d21 + mua*d21_dx)
    dxaxb_dx[0] = +mua - 1
    dxaxb_dx[1] = -mua
    dxaxb_dx[2] = -mub + 1
    dxaxb_dx[3] = +mub
    # dxaxb_dx[0] += + mua - 1
    # dxaxb_dx[1] += - mua
    # dxaxb_dx[2] += - mub + 1
    # dxaxb_dx[3] += + mub

    _d = x3 - x1 + mub*d43 - mua*d21
    dd2_dx = 2 * _d * dxaxb_dx
    print(np.round(mua, 2), np.round(mub, 2))
    return dd2_dx

    # if mua == 0 and mub == 0:
    #     d_d2[0] = x3
    #     d_d2[1] = 0
    #     d_d2[2] = x1
    #     d_d2[3] = 0
    #
    # elif mua == 0 and mub == 1:
    #     d_d2[0] = x4
    #     d_d2[1] = 0
    #     d_d2[2] = 0
    #     d_d2[3] = x1
    #
    # elif mua == 1 and mub == 0:
    #     d_d2[0] = 0
    #     d_d2[1] = x3
    #     d_d2[2] = x2
    #     d_d2[3] = 0
    #
    # elif mua == 1 and mub == 1:
    #     d_d2[0] = 0
    #     d_d2[1] = x4
    #     d_d2[2] = 0
    #     d_d2[3] = x2


def symbolic_jac():
    from sympy import MatrixSymbol, sympify, Symbol, parsing

    x1 = MatrixSymbol('x1', 3, 1)
    x2 = MatrixSymbol('x2', 3, 1)
    x3 = MatrixSymbol('x3', 3, 1)
    x4 = MatrixSymbol('x4', 3, 1)

    d13 = x1 - x3
    d21 = x2 - x1
    d43 = x4 - x3
    d1321 = d13.T * d21
    d1343 = d13.T * d43
    d2121 = d21.T * d21
    d4321 = d43.T * d21
    d4343 = d43.T * d43

    mua = (d1343 * d4321 - d1321 * d4343) / (d2121 * d4343 - d4321 * d4321)
    mub = (d1343 + mua * d4321) / d4343
    mub0 = d1343 / d4343
    mub1 = (d1343 + d4321) / d4343

    xa_0 = x1
    xa_1 = x2
    xa_y = x1 + d21 * mua

    xb_0 = x3
    xb_1 = x4
    xb_y = x3 + d43*mub
    xb_y0 = x3 + d43*mub1
    xb_y1 = x3 + d43*mub0


    d_00 = xa_0.T * xb_0
    d_01 = xa_0.T * xb_1
    d_10 = xa_1.T * xb_0
    d_11 = xa_1.T * xb_1
    d_y1 = xa_y.T * xb_0
    d_y1 = xa_y.T * xb_1
    d_0y = xa_0.T * xb_y0
    d_1y = xa_1.T * xb_y1
    d_yy = xa_y.T * xb_y


    def get_derivative_x(e, x_list, times_den=False):

        d1321_sy = MatrixSymbol('d1321', 1, 1)
        d1343_sy = MatrixSymbol('d1343', 1, 1)
        d2121_sy = MatrixSymbol('d2121', 1, 1)
        d4321_sy = MatrixSymbol('d4321', 1, 1)
        d4343_sy = MatrixSymbol('d4343', 1, 1)
        nom_sy = MatrixSymbol('nom', 1, 1)
        den_sy = MatrixSymbol('den', 1, 1)
        d13_sy = MatrixSymbol('d13', 3, 1)
        d21_sy = MatrixSymbol('d21', 3, 1)
        d43_sy = MatrixSymbol('d43', 3, 1)

        Z1 = (d1343_sy * d4321_sy - d1321_sy * d4343_sy)
        Z2 = (d4321_sy * d1343_sy - d1321_sy * d4343_sy)
        Z3 = (d1343_sy * d4321_sy - d4343_sy * d1321_sy)
        Z4 = (d4321_sy * d1343_sy - d4343_sy * d1321_sy)
        N1 = (d2121_sy * d4343_sy - d4321_sy * d4321_sy)
        N2 = (d4343_sy * d2121_sy - d4321_sy * d4321_sy)


        de_dx_list = []
        for x in x_list:
            de_dx = e.diff(x)
            de_dx = de_dx.subs([(d1321, d1321_sy), (d1321.T, d1321_sy),
                                (d1343, d1343_sy), (d1343.T, d1343_sy),
                                (d2121, d2121_sy), (d2121.T, d2121_sy),
                                (d4321, d4321_sy), (d4321.T, d4321_sy),
                                (d4343, d4343_sy), (d4343.T, d4343_sy)])
            de_dx = de_dx.subs([(Z1, nom_sy),
                                (Z2, nom_sy),
                                (Z3, nom_sy),
                                (Z4, nom_sy),
                                (N1, den_sy),
                                (N2, den_sy)])
            de_dx = de_dx.subs([(d13_sy, d13),
                                (d21_sy, d21),
                                (d43_sy, d43)])

            if times_den:
                de_dx = de_dx*den_sy

            de_dx_list.append(de_dx)

        return de_dx_list


    def final_print(e):
        s = e.__str__()
        s = s.replace('.T', '')
        x1_sy = Symbol('x1')
        x2_sy = Symbol('x2')
        x3_sy = Symbol('x3')
        x4_sy = Symbol('x4')
        d21_sy = Symbol('d21')
        d43_sy = Symbol('d43')
        d13_sy = Symbol('d13')
        e = parsing.parse_expr(s, local_dict={'d1321': Symbol('d1321'),
                                              'd1343': Symbol('d1343'),
                                              'd2121': Symbol('d2121'),
                                              'd4321': Symbol('d4321'),
                                              'd4343': Symbol('d4343'),
                                              'N': Symbol('N'),
                                              'Z': Symbol('Z'),
                                              'x1': x1_sy,
                                              'x2': x2_sy,
                                              'x3': x3_sy,
                                              'x4': x4_sy})

        e = e.subs([(-x1_sy+x2_sy, d21_sy),
                    (2*(-x1_sy+x2_sy), 2*d21_sy),
                    (-x3_sy+x4_sy, d43_sy),
                    (2*(-x3_sy+x4_sy), 2*d43_sy),
                    (+x1_sy-x3_sy, d13_sy),
                    (2*(+x1_sy-x3_sy), 2*d13_sy)])

        return e.expand()


    def final2(s):
        s = s.replace('den**2', 'den2')
        s = s.replace('d4321**2', 'd4321_2')
        s = s.replace('d4343**2', 'd4343_2')
        # s = s.replace('*', ' * ')
        # s = s.replace('/', ' / ')
        # s = s.replace('+', ' + ')
        # s = s.replace('-', ' - ')
        # s = s.replace(' d21 * d4321_2 ', ' d4321_2_d21 ')
        # s = s.replace(' d21 * d4321 ', ' d4321_d21 ')
        # s = s.replace(' d43 * d4321_2 ', ' d4321_2_d43 ')
        # s = s.replace(' d43 * d4321 ', ' d4321_d43 ')
        # s = s.replace(' d21 * d4343 ', ' d4343_d21 ')
        # s = s.replace(' d13 * d4343 ', ' d4343_d13 ')
        # s = s.replace(' d1343 * d43 ', ' d1343_d43 ')
        # s = s.replace(' d1343 * d21 ', ' d1343_d21 ')
        # s = s.replace(' d1321 * d43 ', ' d1321_d43 ')
        # s = s.replace(' d2121 * d43 ', ' d2121_d43 ')
        # s = s.replace(' d13 * d4321 ', ' d4321_d13 ')
        # s = s.replace(' nom / dem ', ' mua ')
        #
        # s = s.replace(' den2 * d4343 ', ' d4343_den2 ')

        # s = s.replace(' * ', '*')
        # s = s.replace(' / ', '/')
        # s = s.replace(' + ', '+')
        # s = s.replace(' - ', '-')
        return parsing.parse_expr(s).simplify().expand()


    d_mua = get_derivative_x(mua, [x1, x2, x3, x4], times_den=True)
    d_mub = get_derivative_x(mub, [x1, x2, x3, x4], times_den=True)
    d_mub0 = get_derivative_x(mub0, [x1, x2, x3, x4])
    d_mub1 = get_derivative_x(mub1, [x1, x2, x3, x4])

    for dmu in d_mub:
        e = final_print(dmu)

        s = final2(e.__str__())
        print(s)
        # print()

